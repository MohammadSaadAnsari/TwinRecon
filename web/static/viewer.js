/* Local WebGL2 viewer for dense points and covariance-aware Gaussian splats. */
class ReconViewer {
  constructor(canvas) {
    this.canvas = canvas;
    this.gl = canvas.getContext('webgl2', {alpha: true, antialias: true, premultipliedAlpha: true});
    this.count = 0;
    this.kind = 'cloud';
    this.yaw = .35;
    this.pitch = -.12;
    this.zoom = 2.8;
    this.drag = false;
    this.splats = null;
    this.needsSort = false;
    this.dirty = true;
    this.sharpness = .55;
    if (!this.gl) {
      document.querySelector('#viewer-kind').textContent = 'WEBGL2 IS UNAVAILABLE';
      return;
    }
    this.setupPrograms();
    canvas.addEventListener('pointerdown', event => {
      this.drag = true;
      this.lastX = event.clientX;
      this.lastY = event.clientY;
      canvas.setPointerCapture(event.pointerId);
    });
    canvas.addEventListener('pointermove', event => {
      if (!this.drag) return;
      this.yaw += (event.clientX - this.lastX) * .006;
      this.pitch = Math.max(-1.45, Math.min(1.45, this.pitch + (event.clientY - this.lastY) * .006));
      this.lastX = event.clientX;
      this.lastY = event.clientY;
      this.needsSort = true;
      this.dirty = true;
    });
    canvas.addEventListener('pointerup', () => { this.drag = false; this.needsSort = true; this.dirty = true; });
    canvas.addEventListener('pointercancel', () => { this.drag = false; this.needsSort = true; this.dirty = true; });
    canvas.addEventListener('wheel', event => {
      event.preventDefault();
      this.zoom = Math.max(1.5, Math.min(8, this.zoom * (1 + event.deltaY * .001)));
      this.dirty = true;
    }, {passive: false});
    document.addEventListener('visibilitychange', () => { this.dirty = true; });
    requestAnimationFrame(() => this.draw());
  }

  program(vertex, fragment) {
    const gl = this.gl;
    const compile = (type, source) => {
      const shader = gl.createShader(type);
      gl.shaderSource(shader, source);
      gl.compileShader(shader);
      if (!gl.getShaderParameter(shader, gl.COMPILE_STATUS)) throw Error(gl.getShaderInfoLog(shader));
      return shader;
    };
    const program = gl.createProgram();
    gl.attachShader(program, compile(gl.VERTEX_SHADER, vertex));
    gl.attachShader(program, compile(gl.FRAGMENT_SHADER, fragment));
    gl.linkProgram(program);
    if (!gl.getProgramParameter(program, gl.LINK_STATUS)) throw Error(gl.getProgramInfoLog(program));
    return program;
  }

  setupPrograms() {
    const gl = this.gl;
    this.pointProgram = this.program(`#version 300 es
      in vec3 aPos; in vec3 aColor; out vec3 vColor;
      uniform float uYaw,uPitch,uZoom,uAspect,uDpr;
      void main(){
        float cy=cos(uYaw),sy=sin(uYaw),cp=cos(uPitch),sp=sin(uPitch);
        vec3 q=vec3(cy*aPos.x+sy*aPos.z,aPos.y,-sy*aPos.x+cy*aPos.z);
        q=vec3(q.x,cp*q.y-sp*q.z,sp*q.y+cp*q.z);
        float depth=uZoom-q.z;
        gl_Position=vec4(q.x*2.3/uAspect,q.y*2.3,depth-1.0,depth);
        gl_PointSize=clamp(1.45*uDpr*sqrt(3.4/depth),1.0*uDpr,2.3*uDpr);
        vColor=aColor;
      }`, `#version 300 es
      precision mediump float; in vec3 vColor; out vec4 outColor;
      void main(){if(length(gl_PointCoord-vec2(.5))>.5)discard;
        outColor=vec4(vColor,1.0);}`);
    this.pointBuffer = gl.createBuffer();
    this.pointVao = gl.createVertexArray();
    gl.bindVertexArray(this.pointVao);
    gl.bindBuffer(gl.ARRAY_BUFFER, this.pointBuffer);
    for (const [name, offset] of [['aPos', 0], ['aColor', 12]]) {
      const location = gl.getAttribLocation(this.pointProgram, name);
      gl.enableVertexAttribArray(location);
      gl.vertexAttribPointer(location, 3, gl.FLOAT, false, 24, offset);
    }
    this.pointUniforms = Object.fromEntries(['uYaw','uPitch','uZoom','uAspect','uDpr'].map(name =>
      [name, gl.getUniformLocation(this.pointProgram, name)]));

    this.splatProgram = this.program(`#version 300 es
      precision highp float;
      in vec3 aPos,aColor,aCovA,aCovB;
      in float aOpacity;
      uniform float uYaw,uPitch,uZoom,uSharpness;
      uniform vec2 uViewport;
      out vec3 vColor;
      out float vOpacity;
      out vec2 vLocal;
      void main(){
        float cy=cos(uYaw),sy=sin(uYaw),cp=cos(uPitch),sp=sin(uPitch);
        mat3 rot=mat3(vec3(cy,sp*sy,-cp*sy),vec3(0.0,cp,sp),vec3(sy,-sp*cy,cp*cy));
        vec3 q=rot*aPos;
        float depth=uZoom-q.z;
        if(depth<0.15){gl_Position=vec4(2.0,2.0,2.0,1.0);vOpacity=0.0;return;}
        mat3 cov=mat3(vec3(aCovA.x,aCovA.y,aCovA.z),
                      vec3(aCovA.y,aCovB.x,aCovB.y),
                      vec3(aCovA.z,aCovB.y,aCovB.z));
        // Tighten the displayed ellipsoid without modifying its exported PLY.
        // At zero detail this exactly matches the original footprint.
        float radiusScale=mix(1.0,0.58,uSharpness);
        mat3 viewCov=rot*cov*transpose(rot)*(radiusScale*radiusScale);
        float focal=uViewport.y*1.15;
        vec3 jx=vec3(focal/depth,0.0,focal*q.x/(depth*depth));
        vec3 jy=vec3(0.0,focal/depth,focal*q.y/(depth*depth));
        float pixelVariance=mix(0.5,0.08,uSharpness);
        float a=dot(jx,viewCov*jx)+pixelVariance;
        float b=dot(jx,viewCov*jy);
        float d=dot(jy,viewCov*jy)+pixelVariance;
        float delta=sqrt(max((a-d)*(a-d)+4.0*b*b,0.0));
        float major=max((a+d+delta)*0.5,pixelVariance);
        float minor=max((a+d-delta)*0.5,pixelVariance);
        vec2 axis=abs(b)>0.00001?normalize(vec2(b,major-a)):(a>=d?vec2(1.0,0.0):vec2(0.0,1.0));
        vec2 perp=vec2(-axis.y,axis.x);
        vec2 corner=vec2((gl_VertexID==1||gl_VertexID==3)?1.0:-1.0,
                         (gl_VertexID>=2)?1.0:-1.0);
        vec2 offset=axis*clamp(2.7*sqrt(major),0.7,48.0)*corner.x+
                    perp*clamp(2.7*sqrt(minor),0.7,48.0)*corner.y;
        vec2 pixels=q.xy*focal/depth+offset;
        gl_Position=vec4(pixels*2.0/uViewport*depth,depth-1.0,depth);
        vColor=aColor;
        vOpacity=aOpacity;
        vLocal=corner*2.7;
      }`, `#version 300 es
      precision mediump float;
      in vec3 vColor; in float vOpacity; in vec2 vLocal;
      out vec4 outColor;
      void main(){
        float alpha=vOpacity*exp(-0.5*dot(vLocal,vLocal));
        if(alpha<0.004)discard;
        outColor=vec4(vColor*alpha,alpha);
      }`);
    this.splatBuffer = gl.createBuffer();
    this.splatVao = gl.createVertexArray();
    gl.bindVertexArray(this.splatVao);
    gl.bindBuffer(gl.ARRAY_BUFFER, this.splatBuffer);
    for (const [name, size, offset] of [['aPos',3,0],['aColor',3,12],['aOpacity',1,24],['aCovA',3,28],['aCovB',3,40]]) {
      const location = gl.getAttribLocation(this.splatProgram, name);
      gl.enableVertexAttribArray(location);
      gl.vertexAttribPointer(location, size, gl.FLOAT, false, 52, offset);
      gl.vertexAttribDivisor(location, 1);
    }
    this.splatUniforms = Object.fromEntries(['uYaw','uPitch','uZoom','uViewport','uSharpness'].map(name =>
      [name, gl.getUniformLocation(this.splatProgram, name)]));
    gl.bindVertexArray(null);
  }

  clear() {
    this.count = 0;
    this.splats = null;
    this.dirty = true;
    document.querySelector('#viewer-empty').classList.remove('hidden');
  }

  setSharpness(value) {
    this.sharpness = Math.max(0, Math.min(1, Number(value) / 100));
    this.dirty = true;
  }

  load(buffer, kind) {
    if (!this.gl) return;
    const data = new Float32Array(buffer);
    const stride = kind === 'splats' ? 13 : 6;
    if (data.length % stride) throw Error('Invalid preview data.');
    this.kind = kind;
    this.count = data.length / stride;
    this.dirty = true;
    if (kind === 'splats') {
      this.splats = data;
      this.needsSort = true;
    } else {
      this.splats = null;
      const gl = this.gl;
      gl.bindBuffer(gl.ARRAY_BUFFER, this.pointBuffer);
      gl.bufferData(gl.ARRAY_BUFFER, data, gl.STATIC_DRAW);
    }
    document.querySelector('#viewer-empty').classList.toggle('hidden', this.count > 0);
  }

  sortSplats() {
    if (!this.splats || !this.needsSort) return;
    const source = this.splats;
    const count = this.count;
    const sy = Math.sin(this.yaw), cy = Math.cos(this.yaw);
    const sp = Math.sin(this.pitch), cp = Math.cos(this.pitch);
    const keys = new Float32Array(count);
    const order = Array.from({length: count}, (_, index) => index);
    for (let i = 0; i < count; i++) {
      const at = i * 13;
      keys[i] = sp * source[at+1] + cp * (-sy * source[at] + cy * source[at+2]);
    }
    order.sort((a, b) => keys[a] - keys[b]);
    const sorted = new Float32Array(source.length);
    for (let i = 0; i < count; i++) {
      const start = order[i] * 13;
      sorted.set(source.subarray(start, start+13), i*13);
    }
    const gl = this.gl;
    gl.bindBuffer(gl.ARRAY_BUFFER, this.splatBuffer);
    gl.bufferData(gl.ARRAY_BUFFER, sorted, gl.DYNAMIC_DRAW);
    this.needsSort = false;
    this.lastSort = performance.now();
  }

  draw() {
    const gl = this.gl;
    if (!gl) return;
    const dpr = Math.min(devicePixelRatio || 1, 2);
    const width = Math.max(1, Math.round(this.canvas.clientWidth * dpr));
    const height = Math.max(1, Math.round(this.canvas.clientHeight * dpr));
    if (this.canvas.width !== width || this.canvas.height !== height) {
      this.canvas.width = width;
      this.canvas.height = height;
      this.dirty = true;
    }
    if (this.kind === 'splats' && !this.dirty) {
      requestAnimationFrame(() => this.draw());
      return;
    }
    gl.viewport(0, 0, width, height);
    gl.clearColor(0, 0, 0, 0);
    gl.clear(gl.COLOR_BUFFER_BIT | gl.DEPTH_BUFFER_BIT);
    if (this.count && document.visibilityState !== 'hidden') {
      if (this.kind === 'splats') {
        if (this.needsSort && (!this.drag || performance.now() - (this.lastSort || 0) > 180)) this.sortSplats();
        gl.useProgram(this.splatProgram);
        gl.bindVertexArray(this.splatVao);
        gl.uniform1f(this.splatUniforms.uYaw, this.yaw);
        gl.uniform1f(this.splatUniforms.uPitch, this.pitch);
        gl.uniform1f(this.splatUniforms.uZoom, this.zoom);
        gl.uniform1f(this.splatUniforms.uSharpness, this.sharpness);
        gl.uniform2f(this.splatUniforms.uViewport, width, height);
        gl.disable(gl.DEPTH_TEST);
        gl.enable(gl.BLEND);
        gl.blendFunc(gl.ONE, gl.ONE_MINUS_SRC_ALPHA);
        gl.drawArraysInstanced(gl.TRIANGLE_STRIP, 0, 4, this.count);
      } else {
        if (!this.drag) this.yaw += .00085;
        gl.useProgram(this.pointProgram);
        gl.bindVertexArray(this.pointVao);
        gl.uniform1f(this.pointUniforms.uYaw, this.yaw);
        gl.uniform1f(this.pointUniforms.uPitch, this.pitch);
        gl.uniform1f(this.pointUniforms.uZoom, this.zoom);
        gl.uniform1f(this.pointUniforms.uAspect, width/height);
        gl.uniform1f(this.pointUniforms.uDpr, dpr);
        gl.enable(gl.DEPTH_TEST);
        gl.disable(gl.BLEND);
        gl.drawArrays(gl.POINTS, 0, this.count);
      }
    }
    this.dirty = false;
    requestAnimationFrame(() => this.draw());
  }
}

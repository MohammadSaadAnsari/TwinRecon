const $ = (s) => document.querySelector(s);
const state = { jobs: [], selected: null, view: 'cloud', previewRequest: 0 };
const fmt = (n) => Number.isFinite(Number(n)) ? Number(n).toLocaleString() : '—';
const escapeHtml = (s) => String(s ?? '').replace(/[&<>"']/g, c => ({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));
let toastTimer;
function toast(message){const el=$('#toast');el.textContent=message;el.classList.add('show');clearTimeout(toastTimer);toastTimer=setTimeout(()=>el.classList.remove('show'),4500)}

const viewer=new ReconViewer($('#scene'));

async function api(url,options){const response=await fetch(url,options);if(!response.ok){let message=`Request failed (${response.status})`;try{message=(await response.json()).error||message}catch{}throw Error(message)}return response}
async function loadSamples(){try{const names=await (await api('/api/samples')).json();for(const name of names){const opt=document.createElement('option');opt.value=name;opt.textContent=name.replace(/_object\.[^.]+$/,'').replace(/_/g,' ').replace(/\b\w/g,c=>c.toUpperCase());$('#sample').append(opt)}}catch(e){toast(e.message)}}
async function loadCapabilities(){try{const caps=await (await api('/api/capabilities')).json();if(!caps.rembg){const option=$('#segmenter option[value="rembg"]');option.disabled=true;option.textContent='AI isolation · install rembg + onnxruntime'}}catch{}}
function statusOf(job){return job.status||(job.outputs?.point_cloud?'complete':'incomplete')}
function renderJobs(){const jobs=state.jobs;$('#job-count').textContent=`${jobs.length} CAPTURE${jobs.length===1?'':'S'}`;$('#jobs').innerHTML=jobs.length?jobs.map((job,i)=>{const status=statusOf(job),name=job.input_video?.split('/').pop()?.replace(/_object\.[^.]+$/,'').replace(/\.[^.]+$/,'').replace(/_/g,' ')||job.job_id;return `<button class="job-card ${state.selected===job.job_id?'selected':''}" data-job="${escapeHtml(job.job_id)}"><span class="job-top"><span>SCAN ${String(jobs.length-i).padStart(2,'0')}</span><em>${escapeHtml(status.toUpperCase())}</em></span><strong>${escapeHtml(name)}</strong><small>${fmt(job.points_count)} POINTS · ${fmt(job.frames_processed)} FRAMES</small></button>`}).join(''):'<div class="no-jobs">No finished scans yet. Your first object can start here.</div>';document.querySelectorAll('[data-job]').forEach(button=>button.onclick=()=>selectJob(button.dataset.job))}
async function refreshJobs(){try{const previous=state.jobs.find(j=>j.job_id===state.selected);const jobs=await (await api('/api/jobs')).json();state.jobs=jobs;renderJobs();if(!state.selected){const first=jobs.find(j=>statusOf(j)==='complete')||jobs[0];if(first)selectJob(first.job_id)}else{const selected=jobs.find(j=>j.job_id===state.selected);if(selected){renderDetails(selected);if(selected.outputs?.point_cloud&&!previous?.outputs?.point_cloud)loadPreview()}}}catch(e){toast(e.message)}}
function artifactLink(id,key,label){return `<a href="/api/jobs/${encodeURIComponent(id)}/files/${key}" download>↗ ${label}</a>`}
function renderDetails(job){
  const id=job.job_id,status=statusOf(job),outs=job.outputs||{},consistency=job.stages?.consistency;
  $('#viewer-title').textContent=(job.input_video?.split('/').pop()||id).replace(/_object\.[^.]+$/,'').replace(/\.[^.]+$/,'').replace(/_/g,' ');
  $('#selected-status').textContent=status.toUpperCase();
  $('#selected-meta').textContent=`${fmt(job.points_count)} POINTS / ${fmt(job.splat_count)} GAUSSIANS`;
  $('#details-id').textContent=id.toUpperCase();
  $('#metric-points').textContent=fmt(job.points_count);
  $('#metric-splats').textContent=fmt(job.splat_count||job.points_count);
  $('#splat-limit option[value="500000"]').textContent=job.splat_count&&job.splat_count<=500000?`ALL ${fmt(job.splat_count)}`:'MAX · UP TO 500K';
  $('#metric-views').textContent=fmt(job.registered_cameras||job.frames_processed);
  $('#metric-weak').textContent=consistency?`${((consistency.mesh_disagreement_fraction??consistency.unreliable_fraction)*100).toFixed(1)}%`:'—';
  $('#metric-explanation').textContent=consistency?
    `${fmt(consistency.unreliable_points)} cloud points are farther than ${Number(consistency.distance_threshold).toPrecision(3)} scene units from the Poisson mesh. This is a consistency diagnostic, not ground-truth accuracy. A reference scan is required to measure accuracy.`:
    'This measures cloud-to-mesh agreement. Ground-truth accuracy needs a reference scan and is not available from one video.';
  $('#artifacts').innerHTML=[['point_cloud','POINT CLOUD .PLY'],['gaussian_splat','GAUSSIAN SPLAT .PLY'],['poisson_mesh','POISSON MESH .PLY'],['consistency_map','CONSISTENCY .PLY'],['cameras','CAMERAS .JSON']].filter(([key])=>outs[key]).map(([key,label])=>artifactLink(id,key,label)).join('');
  const img=$('#seg-preview');
  if(outs.segmentation_preview)img.src=`/api/jobs/${encodeURIComponent(id)}/files/segmentation_preview`;
  else img.removeAttribute('src');
  if(status==='failed')toast(job.error||'Reconstruction failed; inspect the log.');
}
async function selectJob(id){
  state.selected=id;
  viewer.clear();
  renderJobs();
  const job=state.jobs.find(j=>j.job_id===id);
  if(!job)return;
  renderDetails(job);
  await Promise.all([loadPreview(),loadLog()]);
}
async function loadPreview(){
  const id=state.selected,kind=state.view,request=++state.previewRequest;
  if(!id)return;
  $('#render-quality').classList.toggle('visible',kind==='splats');
  $('#viewer-kind').textContent='LOADING PREVIEW…';
  try{
    const query=kind==='splats'?`?limit=${$('#splat-limit').value}`:'';
    const response=await api(`/api/jobs/${encodeURIComponent(id)}/preview/${kind}${query}`);
    const data=await response.arrayBuffer();
    if(request!==state.previewRequest||id!==state.selected||kind!==state.view)return;
    viewer.load(data,kind);
    const label={cloud:'OBJECT-ONLY POINT CLOUD',splats:'GAUSSIAN SPLAT · LOCAL RENDER',consistency:'CLOUD ↔ MESH CONSISTENCY'}[kind];
    $('#viewer-kind').textContent=`${label} · ${fmt(viewer.count)} ${kind==='splats'?'SPLATS':'POINTS'}`;
  }catch(error){
    if(request!==state.previewRequest)return;
    viewer.clear();
    $('#viewer-kind').textContent=`${kind.toUpperCase()} PREVIEW UNAVAILABLE`;
    toast(error.message);
  }
}
async function loadLog(){if(!state.selected)return;try{const text=await (await api(`/api/jobs/${encodeURIComponent(state.selected)}/log`)).text();$('#job-log').textContent=text.slice(-10000)||'No log available yet.';$('#job-log').scrollTop=$('#job-log').scrollHeight}catch{}}
document.querySelectorAll('[data-view]').forEach(button=>button.onclick=()=>{state.view=button.dataset.view;viewer.clear();document.querySelectorAll('[data-view]').forEach(b=>b.classList.toggle('active',b===button));loadPreview()});
$('#splat-limit').onchange=()=>{if(state.view==='splats'){viewer.clear();loadPreview()}};
$('#splat-sharpness').oninput=event=>{
  viewer.setSharpness(event.target.value);
  $('#sharpness-value').textContent=event.target.value;
};
const input=$('#video'),upload=$('#upload-zone'),sample=$('#sample');input.onchange=()=>{if(input.files[0]){sample.value='';$('#upload-title').textContent=input.files[0].name;$('#upload-subtitle').textContent=`${(input.files[0].size/1048576).toFixed(1)} MB · READY TO RECONSTRUCT`}};sample.onchange=()=>{if(sample.value){input.value='';$('#upload-title').textContent='Drop a video here';$('#upload-subtitle').textContent='or click to choose · MP4, MOV, MKV, AVI, WebM'}};
['dragenter','dragover'].forEach(event=>upload.addEventListener(event,e=>{e.preventDefault();upload.classList.add('dragover')}));['dragleave','drop'].forEach(event=>upload.addEventListener(event,e=>{e.preventDefault();upload.classList.remove('dragover')}));upload.addEventListener('drop',e=>{if(e.dataTransfer.files.length){input.files=e.dataTransfer.files;input.dispatchEvent(new Event('change'))}});
$('#job-form').onsubmit=async e=>{e.preventDefault();const button=$('.run-button');button.disabled=true;button.querySelector('span').textContent='STARTING…';try{const form=new FormData(e.currentTarget);if(!input.files.length&&!sample.value)throw Error('Choose a video or a sample to begin.');const job=await (await api('/api/jobs',{method:'POST',body:form})).json();toast('Scan started. The pipeline is working in the background.');state.selected=job.job_id;viewer.clear();setTimeout(()=>refreshJobs(),1200)}catch(error){toast(error.message)}finally{button.disabled=false;button.querySelector('span').textContent='START RECONSTRUCTION'}};
loadSamples();loadCapabilities();refreshJobs();setInterval(async()=>{await refreshJobs();if(state.selected)loadLog()},5000);

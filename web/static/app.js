const $ = (s) => document.querySelector(s);
const state = { jobs: [], selected: null, view: 'cloud', previewRequest: 0, atlasFor: null };
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
async function refreshJobs(){try{const previous=state.jobs.find(j=>j.job_id===state.selected);const jobs=await (await api('/api/jobs')).json();state.jobs=jobs;renderJobs();if(!state.selected){const first=jobs.find(j=>statusOf(j)==='complete')||jobs[0];if(first)selectJob(first.job_id)}else{const selected=jobs.find(j=>j.job_id===state.selected);if(selected){renderDetails(selected);if(selected.outputs?.point_cloud&&!previous?.outputs?.point_cloud)loadPreview();if(selected.outputs?.atlas_data&&state.atlasFor!==selected.job_id)loadAtlas(selected.job_id)}}}catch(e){toast(e.message)}}
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
  const atlas=job.stages?.atlas;
  $('#metric-weak').textContent=atlas?.weak_region_fraction_by_points!==undefined?`${(atlas.weak_region_fraction_by_points*100).toFixed(1)}%`:consistency?`${((consistency.mesh_disagreement_fraction??consistency.unreliable_fraction)*100).toFixed(1)}%`:'—';
  $('#metric-weak').previousElementSibling.textContent=atlas?.weak_region_fraction_by_points!==undefined?'WEAK REGIONS':'MESH GAP';
  $('#metric-explanation').textContent=atlas?.weak_region_fraction_by_points!==undefined?
    `${fmt(atlas.floater_candidates)} floater candidate regions across ${fmt(atlas.sample_points)} sampled points and ${fmt(atlas.views_used?.length)} views. Regional confidence combines camera, mask, image and mesh evidence; it is not ground-truth accuracy.`:
    consistency?`${fmt(consistency.unreliable_points)} cloud points exceed the mesh distance tolerance. This is mesh disagreement, not ground-truth accuracy.`:
    'Regional evidence requires a completed atlas. Ground-truth accuracy requires a reference scan.';
  const refinement=job.stages?.refinement;
  $('#refinement-status').textContent=refinement?.status==='accepted'?
    `SPLAT REFINEMENT ACCEPTED · ${fmt(refinement.optimized_gaussians)} sampled Gaussians · held-out image loss ${Number(refinement.validation_before).toFixed(4)} → ${Number(refinement.validation_final).toFixed(4)} · level-3 SH ${refinement.sh_accepted?'accepted':'held at DC'}`:
    refinement?`SPLAT REFINEMENT ${refinement.status.toUpperCase()} · ${refinement.reason||'Measured MVS colors retained.'}`:
    'SPLAT REFINEMENT · This older scan predates the validation-gated pass.';
  $('#artifacts').innerHTML=[['point_cloud','POINT CLOUD .PLY'],['gaussian_splat','GAUSSIAN SPLAT .PLY'],['poisson_mesh','POISSON MESH .PLY'],['confidence_atlas','CONFIDENCE ATLAS .PLY'],['atlas_data','ATLAS EVIDENCE .JSON'],['consistency_map','MESH GAP .PLY'],['cameras','CAMERAS .JSON']].filter(([key])=>outs[key]).map(([key,label])=>artifactLink(id,key,label)).join('');
  const atlasTab=document.querySelector('[data-view="atlas"]');atlasTab.disabled=!outs.confidence_atlas;
  if(state.view==='atlas'&&!outs.confidence_atlas){state.view='cloud';document.querySelectorAll('[data-view]').forEach(b=>b.classList.toggle('active',b.dataset.view==='cloud'))}
  const img=$('#seg-preview');
  if(outs.segmentation_preview)img.src=`/api/jobs/${encodeURIComponent(id)}/files/segmentation_preview`;
  else img.removeAttribute('src');
  if(status==='failed')toast(job.error||'Reconstruction failed; inspect the log.');
}
async function selectJob(id){
  state.selected=id;
  state.atlasFor=null;
  $('#region-list').innerHTML='<div class="no-evidence">Loading regional evidence…</div>';
  $('#frame-audit').innerHTML='<div class="no-evidence">Loading frame audit…</div>';
  $('#compass-advice').textContent='Loading capture directions…';
  $('#compass-alternates').textContent='';
  $('#compare-result').textContent='';
  viewer.clear();
  renderJobs();
  const job=state.jobs.find(j=>j.job_id===id);
  if(!job)return;
  renderDetails(job);
  await Promise.all([loadPreview(),loadLog(),loadAtlas(id)]);
}
function directionText(item){const turn=item.azimuth_degrees===0?'stay at this azimuth':`rotate ${Math.abs(item.azimuth_degrees)}° ${item.azimuth_degrees>0?'right':'left'}`;const elevation=item.elevation_degrees===0?'at eye level':`${Math.abs(item.elevation_degrees)}° ${item.elevation_degrees>0?'above':'below'} the object`;return `${turn}, ${elevation}`}
async function loadAtlas(id){
  const job=state.jobs.find(j=>j.job_id===id);
  if(!job?.outputs?.atlas_data){$('#region-list').innerHTML='<div class="no-evidence">This scan has no regional atlas. New scans build one automatically.</div>';$('#frame-audit').innerHTML='<div class="no-evidence">This scan has no frame audit.</div>';$('#compass-advice').textContent='Capture advice appears after a new scan.';$('#atlas-summary').textContent='ATLAS NOT AVAILABLE';return}
  try{
    const atlas=await (await api(`/api/jobs/${encodeURIComponent(id)}/atlas`)).json();
    if(state.selected!==id)return;
    state.atlasFor=id;
    $('#atlas-summary').textContent=`${fmt(atlas.regions.length)} REGIONS · ${fmt(atlas.views_used.length)} VIEWS`;
    const regions=atlas.regions.slice(0,6);
    $('#region-list').innerHTML=regions.length?regions.map(region=>`<div class="region-card ${region.floater_candidate?'floater':region.confidence<.8?'weak':''}"><div class="region-card-top"><span>REGION ${region.id}</span><b>${Math.round(region.confidence*100)} / 100</b></div><strong>${escapeHtml(region.reason.toUpperCase())}</strong><small>${fmt(region.points)} points · ${region.camera_support.toFixed(1)} visible views · ${Math.round(region.silhouette_agreement*100)}% mask agreement${region.floater_candidate?' · FLOATER CANDIDATE':''}</small></div>`).join(''):'<div class="no-evidence">No scored regions met the minimum sample size.</div>';
    const flagged=(atlas.frame_audit||[]).filter(frame=>frame.review_reasons?.length);
    $('#frame-audit').innerHTML=flagged.length?flagged.slice(0,4).map(frame=>`<div class="frame-card"><strong>${escapeHtml(frame.image)}</strong><span>${escapeHtml(frame.review_reasons.join(' · '))}</span><small>${Math.round(frame.mask_agreement*100)}% mask agreement · ${frame.point_render_error.toFixed(2)} image error</small></div>`).join(''):'<div class="no-evidence">No sampled frames crossed the audit thresholds.</div>';
    const directions=atlas.capture_compass||[];
    if(directions.length){const first=directions[0];$('#compass-needle').style.transform=`rotate(${first.azimuth_degrees}deg)`;$('#compass-advice').textContent=`${directionText(first)}. Target region ${first.target_region}: ${first.reason}.`;$('#compass-alternates').textContent=directions.slice(1).map((item,i)=>`${i+2}. ${directionText(item)} · relative score ${Math.round(item.relative_gain*100)}%`).join('  /  ')}
    else{$('#compass-advice').textContent='No clear missing direction from the sampled views.';$('#compass-alternates').textContent=''}
    const options=state.jobs.filter(other=>other.job_id!==id&&other.outputs?.atlas_data).map(other=>`<option value="${escapeHtml(other.job_id)}">${escapeHtml((other.input_video?.split('/').pop()||other.job_id).replace(/\.[^.]+$/,''))} · ${escapeHtml(other.job_id)}</option>`);
    $('#compare-job').innerHTML='<option value="">Choose follow-up scan</option>'+options.join('');
  }catch(error){if(state.selected===id){$('#region-list').innerHTML=`<div class="no-evidence">${escapeHtml(error.message)}</div>`;$('#compass-advice').textContent='Capture directions unavailable.'}}
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
    const label={cloud:'OBJECT-ONLY POINT CLOUD',splats:'GAUSSIAN SPLAT · LOCAL RENDER',atlas:'REGIONAL CONFIDENCE ATLAS',consistency:'CLOUD ↔ MESH DISTANCE'}[kind];
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
$('#compare-button').onclick=async()=>{const after=$('#compare-job').value;if(!state.selected||!after){toast('Choose a follow-up scan with an atlas.');return}try{const comparison=await(await api(`/api/compare?before=${encodeURIComponent(state.selected)}&after=${encodeURIComponent(after)}`)).json();const before=comparison.before,now=comparison.after,delta=(now.weak_fraction-before.weak_fraction)*100;$('#compare-result').innerHTML=`WEAK REGIONS ${Math.round(before.weak_fraction*100)}% → ${Math.round(now.weak_fraction*100)}% (${delta>0?'+':''}${delta.toFixed(1)} pp) · CAMERA SUPPORT ${before.mean_camera_support.toFixed(1)} → ${now.mean_camera_support.toFixed(1)} VIEWS · MASK AGREEMENT ${Math.round(before.mean_silhouette_agreement*100)}% → ${Math.round(now.mean_silhouette_agreement*100)}%. <span>${escapeHtml(comparison.note)}</span>`}catch(error){toast(error.message)}};
$('#splat-limit').onchange=()=>{if(state.view==='splats'){viewer.clear();loadPreview()}};
$('#splat-sharpness').oninput=event=>{
  viewer.setSharpness(event.target.value);
  $('#sharpness-value').textContent=event.target.value;
};
const input=$('#video'),upload=$('#upload-zone'),sample=$('#sample');input.onchange=()=>{if(input.files[0]){sample.value='';$('#upload-title').textContent=input.files[0].name;$('#upload-subtitle').textContent=`${(input.files[0].size/1048576).toFixed(1)} MB · READY TO RECONSTRUCT`}};sample.onchange=()=>{if(sample.value){input.value='';$('#upload-title').textContent='Drop a video here';$('#upload-subtitle').textContent='or click to choose · MP4, MOV, MKV, AVI, WebM'}};
['dragenter','dragover'].forEach(event=>upload.addEventListener(event,e=>{e.preventDefault();upload.classList.add('dragover')}));['dragleave','drop'].forEach(event=>upload.addEventListener(event,e=>{e.preventDefault();upload.classList.remove('dragover')}));upload.addEventListener('drop',e=>{if(e.dataTransfer.files.length){input.files=e.dataTransfer.files;input.dispatchEvent(new Event('change'))}});
$('#job-form').onsubmit=async e=>{e.preventDefault();const button=$('.run-button');button.disabled=true;button.querySelector('span').textContent='STARTING…';try{const form=new FormData(e.currentTarget);if(!input.files.length&&!sample.value)throw Error('Choose a video or a sample to begin.');const job=await (await api('/api/jobs',{method:'POST',body:form})).json();toast('Scan started. The pipeline is working in the background.');state.selected=job.job_id;viewer.clear();setTimeout(()=>refreshJobs(),1200)}catch(error){toast(error.message)}finally{button.disabled=false;button.querySelector('span').textContent='START RECONSTRUCTION'}};
loadSamples();loadCapabilities();refreshJobs();setInterval(async()=>{await refreshJobs();if(state.selected)loadLog()},5000);

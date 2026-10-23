/* Each pair shares playback time; each method retains its own duration. */
const data=window.PROJECT;
const modelName=m=>m==='gru1'?'EmbCLIP':'EmbCLIP-GaWF';
const playIcon='<svg viewBox="0 0 24 24" aria-hidden="true"><path d="M5 3l16 9-16 9z"/></svg>';
const pauseIcon='<svg viewBox="0 0 24 24" aria-hidden="true"><path d="M5 3h5v18H5zm9 0h5v18h-5z"/></svg>';
function demo(e){
 const section=document.createElement('article');section.className='demo';
 const heading=document.createElement('h3');heading.textContent=`Find ${e.target}`;
 const titlePlay=document.createElement('button');titlePlay.className='control title-play';titlePlay.type='button';heading.append(titlePlay);section.append(heading);
 const pair=document.createElement('div');pair.className='pair';
 const videos=[],fills=[];
 for(const r of e.models){
  const name=document.createElement('p');name.className='model-name';name.textContent=modelName(r.model);pair.append(name);
  const row=document.createElement('div');row.className='model-row';
  const path=document.createElement('img');path.className='full-path';path.src=r.final_path;path.alt=`${modelName(r.model)} trajectory`;
  const live=document.createElement('div');live.className='live-panel';
  const video=document.createElement('video');video.src=r.asset;video.poster=r.poster;video.preload='metadata';video.muted=true;video.playsInline=true;video.setAttribute('aria-label',`${modelName(r.model)} navigation`);
  const track=document.createElement('div');track.className='timeline';track.setAttribute('aria-hidden','true');
  const fill=document.createElement('div');fill.className='timeline-fill';track.append(fill);live.append(video,track);
  row.append(path,live);pair.append(row);videos.push(video);fills.push(fill);
 }
 const play=document.createElement('button');play.className='control play';play.type='button';play.innerHTML=playIcon;pair.querySelector('.model-row').append(play);section.append(pair);
 const master=e.models[0].steps>=e.models[1].steps?0:1;
 let playing=false,starting=false,raf=0;
 const updateControls=()=>{
  titlePlay.innerHTML=playing?pauseIcon:playIcon;
  for(const button of [play,titlePlay])button.setAttribute('aria-label',`${playing?'Pause':'Play'} both navigation videos for ${e.target}`);
 };
 const updateProgress=()=>videos.forEach((v,i)=>{
  const duration=v.duration;
  const elapsed=Math.max(v.currentTime,videos[master].currentTime);
  const ratio=Number.isFinite(duration)&&duration>0?Math.min(1,elapsed/duration):0;
  fills[i].style.transform=`scaleX(${ratio})`;
 });
 const pause=()=>{playing=false;videos.forEach(v=>v.pause());pair.classList.remove('playing');cancelAnimationFrame(raf);updateControls();updateProgress();};
 const tick=()=>{
  const t=videos[master].currentTime;
  videos.forEach((v,i)=>{if(i!==master&&Number.isFinite(v.duration)){
   const target=Math.min(t,Math.max(0,v.duration-.025));
   if(Math.abs(v.currentTime-target)>.12)v.currentTime=target;
  }});
  updateProgress();
  if(playing&&!videos[master].ended)raf=requestAnimationFrame(tick);else pause();
 };
 const toggle=async()=>{
  if(starting)return;
  if(playing){pause();return;}
  starting=true;
  if(videos[master].ended)videos.forEach(v=>v.currentTime=0);
  try{await Promise.all(videos.map(v=>v.ended?Promise.resolve():v.play()));playing=true;pair.classList.add('playing');play.blur();updateControls();tick();}
  catch{pause();}
  finally{starting=false;}
 };
 pair.addEventListener('click',toggle);titlePlay.addEventListener('click',toggle);
 videos.forEach(v=>{v.addEventListener('loadedmetadata',updateProgress);v.addEventListener('timeupdate',updateProgress);});
 updateControls();return section;
}
[...data.demos].sort((a,b)=>a.target.localeCompare(b.target)||a.episode_id.localeCompare(b.episode_id)).forEach(e=>document.querySelector('#demo-gallery').append(demo(e)));
for(const r of data.attribution||[]){
 const figure=document.createElement('figure');figure.className='attribution';
 if(r.note){const caption=document.createElement('figcaption');caption.textContent=r.note;figure.append(caption);}
 const img=document.createElement('img');img.src=r.asset;img.alt=r.alt||`Final successful target views and gradient sensitivity for ${r.target}`;
 figure.append(img);document.querySelector('#attribution-gallery').append(figure);
}

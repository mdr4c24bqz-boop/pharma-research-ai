"use strict";
self.addEventListener('install',()=>self.skipWaiting());
self.addEventListener('activate',evt=>evt.waitUntil(self.clients.claim()));
self.addEventListener('push',event=>{
 let data={title:'薬学リサーチAI',body:'新着論文があります。',url:'./#watch-notifications'};
 try{if(event.data)data={...data,...event.data.json()};}catch{}
 const url=new URL(data.url||'./#watch-notifications',self.registration.scope);
 if(url.origin!==self.location.origin||!url.pathname.startsWith(new URL(self.registration.scope).pathname))return;
 event.waitUntil(self.registration.showNotification(String(data.title).slice(0,70),{body:String(data.body||'').slice(0,200),icon:'./icon-192.png',badge:'./icon-192.png',tag:'pharma-new-paper',data:{url:url.href}}));
});
self.addEventListener('notificationclick',event=>{
 event.notification.close();
 event.waitUntil((async()=>{
  const url=event.notification.data?.url||new URL('./#watch-notifications',self.registration.scope).href;
  const clients=await self.clients.matchAll({type:'window',includeUncontrolled:true});
  for(const c of clients){if(c.url.startsWith(self.registration.scope)){await c.focus();c.navigate(url);return;}}
  await self.clients.openWindow(url);
 })());
});

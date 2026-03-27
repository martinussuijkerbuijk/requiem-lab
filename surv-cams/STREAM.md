Option A: Just open the browser on the other computer
  The simplest approach — your Flask server already serves over the network. Any
   computer on the same network can open http://192.168.1.x:5000 and get the
  full rendered experience. No extra work needed. The MJPEG stream and WebSocket
   both work from any browser. This is probably the right answer if "the other
  computer" just needs to display it.

  ---
  Option B: NDI
  NDI is a good fit for live production environments (mixing desks, OBS, etc.).
  The flow would be:
  1. Capture the browser canvas as a video stream on the Jetson
  2. Push that as an NDI source
  3. Any NDI receiver (OBS, Resolume, another browser with NDI plugin) picks it
  up

  The challenge is step 1 — getting pixels out of the browser efficiently.
  Options:
  - Use a headless Chrome on the Jetson and capture its framebuffer via
  getDisplayMedia or a native screen capture
  - Or skip the browser entirely and generate the composited frame in Python
  (overlay + video) and push that directly as NDI using ndi-python or PyNDI

  The second approach is cleaner for Jetson: compose the frame + overlay in
  OpenCV/Python and send via NDI. No browser involved on the sending side.

  ---
  Option C: WebRTC
  Low latency (~100ms), works peer-to-peer over a network. The browser already
  has the rendered canvas — you can stream it directly using the WebRTC
  captureStream() API on the canvas element. A signalling server (a few lines of
   Python with aiortc) connects sender and receiver. Works natively in all
  modern browsers on the receiving end.

  Steps:
  1. On sender: canvas.captureStream(30) gives you a MediaStream
  2. Connect it to a WebRTC peer connection
  3. On receiver: display it in a <video> element

  ---
  Option D: Simple RTMP/HLS push
  If the destination is OBS or a streaming platform: capture the canvas stream
  with captureStream() in JS, pipe it through a MediaRecorder to generate
  chunks, send those chunks to a small Python server, which feeds them to FFmpeg
   for RTMP. More moving parts but very compatible with broadcast tools.

  ---
  Recommendation

  - Same LAN, just display it → Option A, already works
  - Into a broadcast/VJ/production tool → Option B (NDI via Python compositing),
   bypasses browser entirely and gives you the cleanest signal
  - Low latency to another browser → Option C (WebRTC), minimal infrastructure
  - Into OBS or streaming platforms → Option D

  What's the receiving end? That determines which path makes most sense.

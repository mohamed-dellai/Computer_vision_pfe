(function(){
  // Shared Crop Editor for reuse across pages
  class CropEditor {
    constructor() {
      this.editor = document.getElementById('crop-editor');
      this.stage = document.getElementById('crop-stage');
      this.image = document.getElementById('crop-editor-img');
      this.frame = document.getElementById('crop-frame');
      this.box = document.getElementById('crop-box');
      this.applyBtn = document.getElementById('apply-crop-btn');
      this.closeBtn = document.getElementById('crop-editor-close');
      this.confirmBtn = document.getElementById('crop-confirm-for-jobs');

      this.cropBox = { left: 0.05, top: 0.05, right: 0.95, bottom: 0.95 };
      this.dragHandle = null;

      this._bound = {
        pointermove: this._onPointerMove.bind(this),
        pointerup: this._onPointerUp.bind(this),
        resize: this._onResize.bind(this)
      };

      this._initEvents();
    }

    _initEvents(){
      if (this.closeBtn) this.closeBtn.addEventListener('click', ()=> this.close());
      if (this.frame) this.frame.addEventListener('pointerdown', (e)=> e.preventDefault());
      document.querySelectorAll('[data-crop-handle]').forEach(h => {
        h.addEventListener('pointerdown', (ev)=> this._startDrag(ev, h.dataset.cropHandle));
      });
      if (this.confirmBtn) this.confirmBtn.addEventListener('click', ()=> this.confirmForJobs());
      if (this.editor) {
        this.editor.addEventListener('pointermove', this._bound.pointermove);
        this.editor.addEventListener('pointerup', this._bound.pointerup);
        this.editor.addEventListener('pointercancel', this._bound.pointerup);
      }
      window.addEventListener('resize', this._bound.resize);
    }

    openWithImage(url){
      return new Promise((resolve, reject) => {
        if (!this.editor || !this.image) return reject(new Error('crop editor DOM missing'));
        this._resolve = resolve;
        this.image.src = url;
        this.image.onload = () => {
          this._ensureValidBox();
          this._render();
          try {
            const imgW = this.image.naturalWidth || this.image.width;
            const imgH = this.image.naturalHeight || this.image.height;
            document.dispatchEvent(new CustomEvent('sharedcrop:opened', { detail: { imageSize: { width: imgW, height: imgH } } }));
          } catch (e) {}
        };
        this.editor.hidden = false;
        document.body.classList.add('crop-editor-open');
      });
    }

    close(){
      if (!this.editor) return;
      this.editor.hidden = true;
      document.body.classList.remove('crop-editor-open');
      if (this._resolve) {
        // closing without confirm -> resolve null
        this._resolve(null);
        this._resolve = null;
      }
    }

    confirmForJobs(){
      if (!this.image || !this._resolve) return;
      const imgW = this.image.naturalWidth || this.image.width;
      const imgH = this.image.naturalHeight || this.image.height;
      const x1 = Math.round(this.cropBox.left * imgW);
      const y1 = Math.round(this.cropBox.top * imgH);
      const x2 = Math.round(this.cropBox.right * imgW);
      const y2 = Math.round(this.cropBox.bottom * imgH);
      const out = [x1, y1, x2, y2];
      const r = this._resolve;
      this._resolve = null;
      this.close();
      r(out);
    }

    _ensureValidBox(){
      this.cropBox.left = clamp(this.cropBox.left, 0, 0.99);
      this.cropBox.top = clamp(this.cropBox.top, 0, 0.99);
      this.cropBox.right = clamp(this.cropBox.right, this.cropBox.left + 0.01, 1);
      this.cropBox.bottom = clamp(this.cropBox.bottom, this.cropBox.top + 0.01, 1);
      this._render();
      try {
        const imgW = this.image ? (this.image.naturalWidth || this.image.width) : null;
        const imgH = this.image ? (this.image.naturalHeight || this.image.height) : null;
        document.dispatchEvent(new CustomEvent('sharedcrop:changed', { detail: { cropBox: this.cropBox, imageSize: { width: imgW, height: imgH } } }));
      } catch (e) {
        // ignore
      }
    }

    _render(){
      if (!this.image || !this.frame || !this.box) return;
      const imageRect = this.image.getBoundingClientRect();
      const stageRect = this.stage.getBoundingClientRect();
      this.frame.style.left = `${imageRect.left - stageRect.left}px`;
      this.frame.style.top = `${imageRect.top - stageRect.top}px`;
      this.frame.style.width = `${imageRect.width}px`;
      this.frame.style.height = `${imageRect.height}px`;
      this.box.style.left = `${this.cropBox.left * 100}%`;
      this.box.style.top = `${this.cropBox.top * 100}%`;
      this.box.style.width = `${(this.cropBox.right - this.cropBox.left) * 100}%`;
      this.box.style.height = `${(this.cropBox.bottom - this.cropBox.top) * 100}%`;
    }

    _startDrag(event, handle){
      event.preventDefault();
      this.dragHandle = handle;
      try { event.currentTarget.setPointerCapture(event.pointerId); } catch {}
      this._updateFromPointer(event);
    }

    _onPointerMove(event){
      if (!this.dragHandle) return;
      this._updateFromPointer(event);
    }

    _onPointerUp(event){
      if (!this.dragHandle) return;
      try { document.releasePointerCapture && document.releasePointerCapture(event.pointerId); } catch {}
      this.dragHandle = null;
    }

    _onResize(){ this._render(); }

    _updateFromPointer(event){
      if (!this.frame || !this.image) return;
      const rect = this.frame.getBoundingClientRect();
      const x = clamp((event.clientX - rect.left) / rect.width, 0, 1);
      const y = clamp((event.clientY - rect.top) / rect.height, 0, 1);
      const minSize = 0.01;
      if (this.dragHandle === 'left') this.cropBox.left = clamp(x, 0, this.cropBox.right - minSize);
      if (this.dragHandle === 'right') this.cropBox.right = clamp(x, this.cropBox.left + minSize, 1);
      if (this.dragHandle === 'top') this.cropBox.top = clamp(y, 0, this.cropBox.bottom - minSize);
      if (this.dragHandle === 'bottom') this.cropBox.bottom = clamp(y, this.cropBox.top + minSize, 1);
      this._render();
    }
  }

  function clamp(v, a, b){ return Math.min(b, Math.max(a, v)); }

  // singleton
  let instance = null;
  function getInstance(){ if (!instance) instance = new CropEditor(); return instance; }

  window.SharedCropEditor = {
    openWithImage: (url) => getInstance().openWithImage(url),
    confirmForJobs: () => getInstance().confirmForJobs(),
    close: () => getInstance().close(),
    render: () => getInstance()._render(),
    getNormalizedCropBox: () => {
      const inst = getInstance();
      return [inst.cropBox.left, inst.cropBox.top, inst.cropBox.right, inst.cropBox.bottom];
    },
    getImageNaturalSize: () => {
      const inst = getInstance();
      if (!inst.image) return null;
      return { width: inst.image.naturalWidth || inst.image.width, height: inst.image.naturalHeight || inst.image.height };
    },
    setNormalizedCropBox: (box) => {
      const inst = getInstance();
      if (!box || box.length !== 4) return;
      inst.cropBox.left = box[0]; inst.cropBox.top = box[1]; inst.cropBox.right = box[2]; inst.cropBox.bottom = box[3];
      inst._ensureValidBox();
    }
  };

})();

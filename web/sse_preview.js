// Comfy-SSE clip previews: a live, scrubbable player on SSE Load Clip (frames
// served by routes.py, in the node's output colourspace) and on SSE Write
// Clip (the H.264 preview written when the node executes).
import { app } from "../../scripts/app.js";
import { api } from "../../scripts/api.js";

const CONTROLS_HEIGHT = 30;

const STYLE = `
.sse-player { display: flex; flex-direction: column; gap: 4px; width: 100%; height: 100%; font: 11px sans-serif; color: #ddd; }
.sse-view { position: relative; flex: 1; min-height: 0; background: #000; border-radius: 4px; overflow: hidden; cursor: pointer; }
.sse-view canvas, .sse-view video { position: absolute; inset: 0; width: 100%; height: 100%; object-fit: contain; }
.sse-status { position: absolute; left: 6px; top: 4px; right: 6px; color: #bbb; text-shadow: 0 0 3px #000; pointer-events: none; overflow-wrap: anywhere; }
.sse-controls { display: flex; align-items: center; gap: 2px; height: 24px; }
.sse-controls button { background: none; border: none; color: inherit; cursor: pointer; padding: 0 5px; font-size: 13px; }
.sse-controls button:disabled { opacity: 0.35; cursor: default; }
.sse-controls input { flex: 1; min-width: 30px; margin: 0 4px; }
.sse-frame { min-width: 64px; text-align: right; color: #999; font-variant-numeric: tabular-nums; }
`;

function el(tag, className, parent) {
    const e = document.createElement(tag);
    if (className) e.className = className;
    parent?.append(e);
    return e;
}

function chain(object, name, fn) {
    const orig = object[name];
    object[name] = function () {
        const r = orig?.apply(this, arguments);
        fn.apply(this, arguments);
        return r;
    };
}

function fitHeight(node) {
    node.setSize([node.size[0], node.computeSize()[1]]);
    node.setDirtyCanvas(true, true);
}

class ClipPlayer {
    constructor(onAspect) {
        this.onAspect = onAspect;
        this.aspect = 0;
        this.gen = 0;
        this.drawToken = 0;
        this.raf = 0;

        this.root = el("div", "sse-player");
        this.view = el("div", "sse-view", this.root);
        this.canvas = el("canvas", "", this.view);
        this.video = el("video", "", this.view);
        this.video.muted = true;
        this.video.loop = true;
        this.video.playsInline = true;
        this.status = el("div", "sse-status", this.view);
        this.view.addEventListener("click", () => this.toggle());

        const bar = el("div", "sse-controls", this.root);
        this.prevBtn = this.button(bar, "◂", "Previous frame", () => this.step(-1));
        this.playBtn = this.button(bar, "▶", "Play / pause", () => this.toggle());
        this.nextBtn = this.button(bar, "▸", "Next frame", () => this.step(1));
        this.scrub = el("input", "", bar);
        this.scrub.type = "range";
        this.scrub.min = 0;
        this.scrub.step = 1;
        this.scrub.addEventListener("input", () => {
            this.pause();
            this.seek(Number(this.scrub.value));
        });
        this.counter = el("span", "sse-frame", bar);

        this.video.addEventListener("loadedmetadata", () => {
            if (this.mode !== "video") return;
            this.count = Math.max(1, Math.round(this.video.duration * this.fps));
            this.setAspect(this.video.videoWidth / this.video.videoHeight);
            this.status.textContent = "";
            this.seek(0);
        });
        this.video.addEventListener("error", () => {
            if (this.mode === "video") this.status.textContent = "preview failed to load";
        });
        this.clear("");
    }

    button(parent, text, title, onClick) {
        const b = el("button", "", parent);
        b.textContent = text;
        b.title = title;
        b.addEventListener("click", onClick);
        return b;
    }

    minHeight(width) {
        return CONTROLS_HEIGHT + (width - 20) / (this.aspect || 16 / 9);
    }

    setAspect(aspect) {
        if (!aspect || Math.abs(aspect - this.aspect) < 0.01) return;
        this.aspect = aspect;
        this.onAspect();
    }

    clear(message) {
        this.pause();
        this.gen++;
        this.mode = null;
        this.key = null;
        this.cache = new Map();
        this.inflight = new Set();
        this.indices = [];
        this.count = 0;
        this.pos = 0;
        this.video.removeAttribute("src");
        this.video.load();
        this.video.style.display = "none";
        this.canvas.style.display = "none";
        this.status.textContent = message;
        this.updateControls();
    }

    // Frames mode: `indices` are source frame numbers, fetched lazily and kept
    // per `key` so range edits reuse what is already cached.
    setFrames({ key, indices, fps, concurrency, fetchFrame }) {
        if (this.mode !== "frames" || key !== this.key) {
            this.clear("");
            this.mode = "frames";
            this.key = key;
        }
        this.indices = indices;
        this.count = indices.length;
        this.canvas.style.display = this.count ? "" : "none";
        this.fps = fps;
        this.concurrency = concurrency;
        this.fetchFrame = fetchFrame;
        this.pos = Math.min(this.pos, Math.max(0, this.count - 1));
        this.updateControls();
        this.show();
        this.pump();
    }

    setVideo(url, fps) {
        this.clear("loading…");
        this.mode = "video";
        this.fps = fps || 24;
        this.video.style.display = "";
        this.video.src = url;
    }

    pump() {
        if (this.mode !== "frames") return;
        while (this.inflight.size < this.concurrency) {
            const src = this.nextMissing();
            if (src === undefined) break;
            this.fetch(src);
        }
        this.updateStatus();
    }

    // nearest uncached frame at or after the playhead, so scrubbing and
    // playback pull the frames needed next
    nextMissing() {
        for (let k = 0; k < this.count; k++) {
            const src = this.indices[(this.pos + k) % this.count];
            if (!this.cache.has(src) && !this.inflight.has(src)) return src;
        }
    }

    fetch(src) {
        const gen = this.gen;
        this.inflight.add(src);
        this.fetchFrame(src).catch(() => false).then((blob) => {
            if (gen !== this.gen) return;
            this.inflight.delete(src);
            this.cache.set(src, blob);
            if (src === this.indices[this.pos]) this.show();
            this.pump();
        });
    }

    updateStatus() {
        if (!this.count) {
            this.status.textContent = "no frames in range";
            return;
        }
        let cached = 0;
        let failed = 0;
        for (const src of this.indices) {
            const blob = this.cache.get(src);
            if (blob !== undefined) cached++;
            if (blob === false) failed++;
        }
        const parts = [];
        if (cached < this.count) parts.push(`caching ${cached}/${this.count}`);
        if (failed) parts.push(`${failed} frame${failed > 1 ? "s" : ""} failed to load`);
        this.status.textContent = parts.join(" · ");
    }

    show() {
        this.updateControls();
        const blob = this.cache.get(this.indices[this.pos]);
        if (!blob) return;
        const token = ++this.drawToken;
        createImageBitmap(blob).then((bmp) => {
            if (token !== this.drawToken) return bmp.close();
            if (this.canvas.width !== bmp.width || this.canvas.height !== bmp.height) {
                this.canvas.width = bmp.width;
                this.canvas.height = bmp.height;
            }
            this.canvas.getContext("2d").drawImage(bmp, 0, 0);
            bmp.close();
            this.setAspect(this.canvas.width / this.canvas.height);
        }, () => {});
    }

    seek(pos) {
        if (!this.count) return;
        this.pos = Math.max(0, Math.min(this.count - 1, pos));
        if (this.mode === "video") {
            this.video.currentTime = (this.pos + 0.5) / this.fps;
            this.updateControls();
        } else {
            this.show();
            this.pump();
        }
    }

    step(delta) {
        this.pause();
        this.seek((this.pos + delta + this.count) % this.count);
    }

    toggle() {
        if (this.playing) this.pause();
        else this.play();
    }

    play() {
        if (this.count < 2) return;
        this.playing = true;
        this.playBtn.textContent = "❚❚";
        if (this.mode === "video") this.video.play().catch(() => {});
        this.last = performance.now();
        if (!this.raf) this.raf = requestAnimationFrame((t) => this.tick(t));
    }

    pause() {
        this.playing = false;
        this.playBtn.textContent = "▶";
        if (this.mode === "video") this.video.pause();
    }

    tick(now) {
        if (!this.playing) {
            this.raf = 0;
            return;
        }
        if (this.mode === "video") {
            const pos = Math.min(this.count - 1, Math.floor(this.video.currentTime * this.fps));
            if (pos !== this.pos) {
                this.pos = pos;
                this.updateControls();
            }
        } else {
            const frameTime = 1000 / this.fps;
            const next = (this.pos + 1) % this.count;
            // hold on the current frame while the next one is still caching
            if (now - this.last >= frameTime && this.cache.has(this.indices[next])) {
                this.last = now - this.last > 2 * frameTime ? now : this.last + frameTime;
                this.pos = next;
                this.show();
            }
        }
        this.raf = requestAnimationFrame((t) => this.tick(t));
    }

    updateControls() {
        const n = this.count;
        this.scrub.max = Math.max(0, n - 1);
        this.scrub.value = this.pos;
        for (const c of [this.scrub, this.prevBtn, this.playBtn, this.nextBtn]) c.disabled = n < 2;
        this.counter.textContent = n ? `${this.pos + 1} / ${n}` : "";
        this.counter.title = this.mode === "frames" && n ? `source frame ${this.indices[this.pos]}` : "";
    }
}

function addPlayer(node) {
    const player = new ClipPlayer(() => fitHeight(node));
    const widget = node.addDOMWidget("sse_preview", "sse_preview", player.root, {
        serialize: false,
        hideOnZoom: false,
        getMinHeight: () => player.minHeight(node.size[0]),
    });
    widget.serialize = false;
    chain(node, "onRemoved", () => player.clear(""));
    return { player, widget };
}

// Draws "value / max" with the max in the label's grey, mirroring
// BaseSteppedWidget.drawWidget's layout.
function addMaxHint(widget) {
    const draw = widget.drawWidget;
    if (!draw) return;
    widget.drawWidget = function (ctx, options) {
        if (this.sseMax == null || !options.showText || this.computedDisabled) return draw.call(this, ctx, options);
        const { fillStyle, strokeStyle, textAlign } = ctx;
        this.drawWidgetShape(ctx, options);
        this.drawArrowButtons(ctx, options.width);
        const y = this.labelBaseline;
        const right = options.width - 50;
        const hint = ` / ${this.sseMax}`;
        ctx.fillStyle = this.secondary_text_color;
        ctx.textAlign = "left";
        ctx.fillText(this.displayName, 35, y);
        ctx.textAlign = "right";
        ctx.fillText(hint, right, y);
        ctx.fillStyle = this.text_color;
        ctx.fillText(this._displayValue, right - ctx.measureText(hint).width, y);
        Object.assign(ctx, { fillStyle, strokeStyle, textAlign });
    };
}

function setupLoadClip(node) {
    const { player, widget } = addPlayer(node);
    const w = (name) => node.widgets.find((x) => x.name === name);
    const get = (name) => w(name)?.value;
    const limit = w("frame_limit");
    addMaxHint(limit);

    let info = null;
    let infoPath = null;
    let infoToken = 0;
    let timer = 0;

    function setMax(max) {
        limit.options.max = max ?? 2 ** 31;
        limit.sseMax = max;
        if (max != null && limit.value > max) limit.value = max;
        node.setDirtyCanvas(true);
    }

    function apply() {
        const show = get("show_preview") !== false;
        if (!!widget.hidden === show) {
            widget.hidden = !show;
            fitHeight(node);
        }
        if (!show) player.pause();
        if (!info) return;

        const start = get("start_frame");
        const nth = Math.max(1, get("every_nth"));
        const available = Math.max(0, Math.ceil((info.frame_count - start) / nth));
        setMax(available);
        if (!show) return;

        const lim = get("frame_limit");
        const n = lim > 0 ? Math.min(lim, available) : available;
        const path = infoPath;
        const ics = get("input_colorspace");
        const ocs = get("output_colorspace");
        const fpsOverride = get("fps_override");
        player.setFrames({
            key: `${path}|${ics}|${ocs}`,
            indices: Array.from({ length: n }, (_, k) => start + k * nth),
            fps: fpsOverride > 0 ? fpsOverride : info.fps,
            concurrency: info.kind === "video" ? 1 : 4,
            fetchFrame: async (index) => {
                const q = new URLSearchParams({ path, index, input_colorspace: ics, output_colorspace: ocs });
                const res = await api.fetchApi(`/sse/clip_frame?${q}`);
                if (!res.ok) throw new Error(await res.text());
                return res.blob();
            },
        });
    }

    async function loadInfo(path) {
        const token = ++infoToken;
        infoPath = path;
        info = null;
        setMax(null);
        player.clear(path ? "reading source…" : "no source");
        if (!path) return;
        let res = null;
        let body = {};
        try {
            res = await api.fetchApi(`/sse/clip_info?${new URLSearchParams({ path })}`);
            body = await res.json();
        } catch {}
        if (token !== infoToken) return;
        if (!res?.ok) {
            player.clear(body.error ?? "could not read source");
            return;
        }
        info = body;
        apply();
    }

    function refresh() {
        const path = String(get("source_path") ?? "").trim();
        if (path !== infoPath) loadInfo(path);
        else apply();
    }

    for (const name of ["start_frame", "frame_limit", "every_nth", "input_colorspace", "output_colorspace", "fps_override", "show_preview"]) {
        const x = w(name);
        if (x) chain(x, "callback", apply);
    }
    chain(w("source_path"), "callback", () => {
        clearTimeout(timer);
        timer = setTimeout(refresh, 300);
    });
    chain(node, "onConfigure", refresh);
    requestAnimationFrame(refresh);
}

function setupWriteClip(node) {
    const { player, widget } = addPlayer(node);
    widget.hidden = true;
    chain(node, "onExecuted", (message) => {
        const clip = message?.gifs?.[0];
        if (!clip) return;
        const q = new URLSearchParams({ filename: clip.filename, subfolder: clip.subfolder, type: clip.type });
        widget.hidden = false;
        player.setVideo(api.apiURL(`/view?${q}`), clip.frame_rate);
        fitHeight(node);
    });
}

app.registerExtension({
    name: "SSE.ClipPreview",
    setup() {
        el("style", "", document.head).textContent = STYLE;
    },
    beforeRegisterNodeDef(nodeType, nodeData) {
        if (nodeData.name === "SSE_LoadClip") {
            chain(nodeType.prototype, "onNodeCreated", function () {
                setupLoadClip(this);
            });
        } else if (nodeData.name === "SSE_WriteClip") {
            chain(nodeType.prototype, "onNodeCreated", function () {
                setupWriteClip(this);
            });
        }
    },
});

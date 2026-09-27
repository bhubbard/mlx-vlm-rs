// Architecture Specs Data
const ARCH_DATA = {
  paligemma: {
    name: "PaliGemma 3B (224px)",
    params: "2.92B",
    vision: {
      backbone: "SigLIP-So400m",
      imageSize: "224x224",
      patchSize: "14x14",
      patches: 256,
      dim: 1152,
      layers: 27,
      heads: 16
    },
    language: {
      backbone: "Gemma 2B Causal LM",
      hiddenDim: 2048,
      layers: 18,
      heads: 8,
      intermediate: 16384,
      vocabSize: 257216,
      imageTokenId: 257152
    }
  },
  qwen2vl: {
    name: "Qwen2-VL 2B Instruct",
    params: "2.21B",
    vision: {
      backbone: "Dynamic ViT",
      imageSize: "448x448",
      patchSize: "14x14",
      patches: 1024,
      dim: 1280,
      layers: 32,
      heads: 16
    },
    language: {
      backbone: "Qwen2 Causal LM",
      hiddenDim: 1536,
      layers: 28,
      heads: 12,
      intermediate: 8960,
      vocabSize: 152064,
      imageTokenId: 151655
    }
  },
  smolvlm: {
    name: "SmolVLM 500M Instruct",
    params: "450M",
    vision: {
      backbone: "SigLIP ViT",
      imageSize: "384x384",
      patchSize: "16x16",
      patches: 576,
      dim: 768,
      layers: 12,
      heads: 12
    },
    language: {
      backbone: "SmolLM-360M",
      hiddenDim: 960,
      layers: 16,
      heads: 15,
      intermediate: 2560,
      vocabSize: 49152,
      imageTokenId: 49150
    }
  },
  llava: {
    name: "LLaVA 1.5 7B",
    params: "7.06B",
    vision: {
      backbone: "CLIP ViT-L/14",
      imageSize: "336x336",
      patchSize: "14x14",
      patches: 576,
      dim: 1024,
      layers: 24,
      heads: 16
    },
    language: {
      backbone: "Llama 2 7B",
      hiddenDim: 4096,
      layers: 32,
      heads: 32,
      intermediate: 11008,
      vocabSize: 32000,
      imageTokenId: 32000
    }
  }
};

// Simulation presets
const SIM_PRESETS = {
  chart: {
    icon: "📊",
    title: "Financial Infographic (Q3 ARR Growth)",
    prompt: "Extract the Q3 revenue metrics and summarize the YoY growth trend.",
    response: "Based on the chart:\n• Q3 Revenue reached $42.8M, representing a +34.2% YoY increase compared to $31.9M in Q3 prior year.\n• Net Retention Rate remains robust at 118%.\n• Gross margin expanded by 140 bps to 78.4%."
  },
  receipt: {
    icon: "🧾",
    title: "Store Receipt (Supermarket Checkout)",
    prompt: "Itemize all purchased groceries and compute the subtotal and tax.",
    response: "Receipt breakdown:\n1. Organic Honeycrisp Apples: $4.99\n2. Oat Milk 64oz: $3.89\n3. Sourdough Loaf: $5.25\n4. Greek Yogurt 32oz: $4.50\n---\nSubtotal: $18.63\nSales Tax (8.25%): $1.54\nTotal Paid: $20.17"
  },
  wildlife: {
    icon: "🦅",
    title: "High-Resolution Wildlife Photo",
    prompt: "Identify the bird species, habitat, and notable plumage features.",
    response: "This is an adult Bald Eagle (Haliaeetus leucocephalus):\n• Distinct white head and tail contrasting with dark brown body plumage.\n• Large, sharply hooked yellow beak and piercing yellow irises.\n• Native to North American riparian zones and coastal estuaries."
  },
  code: {
    icon: "💻",
    title: "UI Mockup Screenshot",
    prompt: "Generate clean Rust Leptos / Tailwind markup for this card component.",
    response: "```rust\nview! {\n  <div class=\"p-6 rounded-2xl bg-slate-900 border border-slate-800 shadow-xl\">\n    <h3 class=\"text-lg font-bold text-sky-400\">\"Unified Memory MLX\"</h3>\n    <p class=\"mt-2 text-sm text-slate-400\">\"Zero-copy tensor transfers directly to Apple Silicon GPU.\"</p>\n  </div>\n}\n```"
  }
};

document.addEventListener("DOMContentLoaded", () => {
  initArchTabs();
  initPatchGrid();
  initTokenStream();
  initSimulator();
});

function initArchTabs() {
  const tabs = document.querySelectorAll(".arch-tab");
  const display = document.getElementById("arch-display");

  function renderArch(key) {
    const arch = ARCH_DATA[key];
    display.innerHTML = `
      <div class="spec-group">
        <h4>Overview</h4>
        <div class="spec-item"><span>Model</span><span>${arch.name}</span></div>
        <div class="spec-item"><span>Parameters</span><span>${arch.params}</span></div>
        <div class="spec-item"><span>Precision</span><span>BF16 / FP16</span></div>
      </div>
      <div class="spec-group">
        <h4>Vision Backbone</h4>
        <div class="spec-item"><span>Architecture</span><span>${arch.vision.backbone}</span></div>
        <div class="spec-item"><span>Input Resolution</span><span>${arch.vision.imageSize}</span></div>
        <div class="spec-item"><span>Patch Size</span><span>${arch.vision.patchSize}</span></div>
        <div class="spec-item"><span>Patches</span><span>${arch.vision.patches}</span></div>
        <div class="spec-item"><span>Vision Dim</span><span>${arch.vision.dim}</span></div>
        <div class="spec-item"><span>ViT Layers</span><span>${arch.vision.layers} (${arch.vision.heads} heads)</span></div>
      </div>
      <div class="spec-group">
        <h4>Language Backbone</h4>
        <div class="spec-item"><span>LM Backbone</span><span>${arch.language.backbone}</span></div>
        <div class="spec-item"><span>Hidden Dim</span><span>${arch.language.hiddenDim}</span></div>
        <div class="spec-item"><span>LM Layers</span><span>${arch.language.layers} (${arch.language.heads} heads)</span></div>
        <div class="spec-item"><span>Intermediate Dim</span><span>${arch.language.intermediate}</span></div>
        <div class="spec-item"><span>Vocabulary Size</span><span>${arch.language.vocabSize}</span></div>
        <div class="spec-item"><span>&lt;image&gt; Token ID</span><span>${arch.language.imageTokenId}</span></div>
      </div>
    `;
  }

  tabs.forEach(tab => {
    tab.addEventListener("click", () => {
      tabs.forEach(t => t.classList.remove("active"));
      tab.classList.add("active");
      renderArch(tab.dataset.arch);
    });
  });

  renderArch("paligemma");
}

function initPatchGrid() {
  const grid = document.getElementById("patch-grid");
  grid.innerHTML = "";
  for (let i = 0; i < 16; i++) {
    const cell = document.createElement("div");
    cell.className = "patch-cell";
    grid.appendChild(cell);
  }
}

function initTokenStream() {
  const stream = document.getElementById("token-stream");
  const tokens = [
    { type: "text", text: "&lt;bos&gt;" },
    { type: "vision", text: "v_tok_1" },
    { type: "vision", text: "v_tok_2" },
    { type: "vision", text: "..." },
    { type: "vision", text: "v_tok_256" },
    { type: "text", text: "Describe" },
    { type: "text", text: "this" },
    { type: "text", text: "image" },
    { type: "text", text: "&lt;eos&gt;" }
  ];

  tokens.forEach(tok => {
    const pill = document.createElement("span");
    pill.className = `token-pill ${tok.type}`;
    pill.innerHTML = tok.text;
    stream.appendChild(pill);
  });
}

function initSimulator() {
  const presetSelect = document.getElementById("sim-preset");
  const tempSlider = document.getElementById("temp-slider");
  const tempVal = document.getElementById("temp-val");
  const tokensSlider = document.getElementById("tokens-slider");
  const tokensVal = document.getElementById("tokens-val");
  const promptInput = document.getElementById("sim-prompt");
  const imgDisplay = document.getElementById("sim-img-display");
  const runBtn = document.getElementById("run-sim-btn");
  const output = document.getElementById("sim-output");
  const metrics = document.getElementById("gen-metrics");

  tempSlider.addEventListener("input", e => tempVal.textContent = e.target.value);
  tokensSlider.addEventListener("input", e => tokensVal.textContent = e.target.value);

  function loadPreset(key) {
    const p = SIM_PRESETS[key];
    promptInput.value = p.prompt;
    imgDisplay.innerHTML = `<span style="font-size: 2.5rem;">${p.icon}</span><span>${p.title}</span>`;
  }

  presetSelect.addEventListener("change", e => loadPreset(e.target.value));
  loadPreset("chart");

  let typingTimer = null;

  runBtn.addEventListener("click", () => {
    if (typingTimer) clearInterval(typingTimer);

    const preset = SIM_PRESETS[presetSelect.value];
    const fullText = preset.response;
    output.textContent = "";
    runBtn.disabled = true;
    runBtn.textContent = "Streaming tokens...";

    metrics.textContent = "Prefill: 1.2ms (256 visual tokens)";

    let index = 0;
    const words = fullText.split(" ");
    const startTime = performance.now();

    typingTimer = setInterval(() => {
      if (index < words.length) {
        output.textContent += (index === 0 ? "" : " ") + words[index];
        index++;
        const elapsed = (performance.now() - startTime) / 1000;
        const tokPerSec = (index / Math.max(0.001, elapsed)).toFixed(1);
        metrics.textContent = `Streaming: ${index} tokens | ${tokPerSec} tok/s`;
      } else {
        clearInterval(typingTimer);
        runBtn.disabled = false;
        runBtn.textContent = "Generate Multi-Modal Response";
        const totalElapsed = ((performance.now() - startTime) / 1000).toFixed(2);
        metrics.textContent = `Complete: ${words.length} tokens in ${totalElapsed}s (128 tok/s decode)`;
      }
    }, 28);
  });
}

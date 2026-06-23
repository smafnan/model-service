import { useEffect, useRef, useState } from "react";
import { motion } from "framer-motion";
import { Activity, Smile, Meh, Frown, Sparkles } from "lucide-react";

type Pred = { label: string; score: number; scores: Record<string, number> };

const ORDER = ["positive", "neutral", "negative"];
const STYLE: Record<string, { bar: string; text: string; icon: any }> = {
  positive: { bar: "from-emerald-400 to-teal-400", text: "text-emerald-300", icon: Smile },
  neutral: { bar: "from-slate-400 to-slate-500", text: "text-slate-300", icon: Meh },
  negative: { bar: "from-rose-400 to-pink-500", text: "text-rose-300", icon: Frown },
};

const EXAMPLES = [
  "Absolutely love this, best purchase of the year!",
  "It's okay, nothing special but does the job.",
  "Terrible quality, broke after one day. Avoid.",
];

function Blobs() {
  return (
    <div className="pointer-events-none fixed inset-0 overflow-hidden">
      <div className="absolute -top-44 left-1/4 h-[34rem] w-[34rem] rounded-full bg-indigo-600/20 blur-3xl animate-float" />
      <div className="absolute bottom-0 -right-40 h-[30rem] w-[30rem] rounded-full bg-emerald-500/12 blur-3xl animate-float [animation-delay:-6s]" />
      <div className="absolute -bottom-40 -left-32 h-[28rem] w-[28rem] rounded-full bg-rose-500/12 blur-3xl animate-float [animation-delay:-9s]" />
    </div>
  );
}

export default function App() {
  const [text, setText] = useState(EXAMPLES[0]);
  const [pred, setPred] = useState<Pred | null>(null);
  const [healthy, setHealthy] = useState<boolean | null>(null);
  const timer = useRef<number | null>(null);

  useEffect(() => {
    fetch("/health").then((r) => r.json()).then((d) => setHealthy(d.model_loaded)).catch(() => setHealthy(false));
  }, []);

  // Live classify, debounced.
  useEffect(() => {
    if (timer.current) clearTimeout(timer.current);
    if (!text.trim()) {
      setPred(null);
      return;
    }
    timer.current = window.setTimeout(async () => {
      try {
        const r = await fetch("/predict", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ text }),
        });
        if (r.ok) setPred(await r.json());
      } catch {
        /* ignore */
      }
    }, 350);
    return () => {
      if (timer.current) clearTimeout(timer.current);
    };
  }, [text]);

  const WinIcon = pred ? STYLE[pred.label].icon : Sparkles;

  return (
    <div className="relative min-h-screen text-slate-200">
      <Blobs />
      <div className="relative mx-auto max-w-3xl px-5 py-14">
        <motion.header
          initial={{ opacity: 0, y: -16 }}
          animate={{ opacity: 1, y: 0 }}
          className="mb-9 text-center"
        >
          <div className="mb-4 inline-flex items-center gap-2 rounded-full border border-white/10 bg-white/5 px-4 py-1.5 text-xs text-slate-300 backdrop-blur">
            <Activity size={14} className={healthy ? "text-emerald-400" : "text-amber-400"} />
            {healthy === null ? "Connecting…" : healthy ? "Model loaded · live" : "Model unavailable"}
          </div>
          <h1 className="text-5xl font-extrabold tracking-tight text-white sm:text-6xl">
            Sentiment{" "}
            <span className="bg-gradient-to-r from-emerald-300 via-indigo-300 to-rose-300 bg-clip-text text-transparent">
              Playground
            </span>
          </h1>
          <p className="mx-auto mt-4 max-w-lg text-slate-400">
            Type anything — it classifies in real time as positive, neutral, or
            negative, served by the containerised FastAPI model.
          </p>
        </motion.header>

        {/* Input */}
        <motion.div
          initial={{ opacity: 0, y: 16 }}
          animate={{ opacity: 1, y: 0 }}
          className="rounded-2xl border border-white/10 bg-white/[0.04] p-4 backdrop-blur"
        >
          <textarea
            value={text}
            onChange={(e) => setText(e.target.value)}
            placeholder="Write a review, a tweet, anything…"
            className="h-28 w-full resize-none bg-transparent text-lg text-white placeholder:text-slate-600 outline-none"
          />
        </motion.div>

        <div className="mt-3 flex flex-wrap justify-center gap-2">
          {EXAMPLES.map((e) => (
            <button
              key={e}
              onClick={() => setText(e)}
              className="rounded-full border border-white/10 bg-white/5 px-3 py-1 text-xs text-slate-400 hover:bg-white/10"
            >
              {e.length > 38 ? e.slice(0, 38) + "…" : e}
            </button>
          ))}
        </div>

        {/* Prediction */}
        {pred && (
          <motion.div
            initial={{ opacity: 0, y: 16 }}
            animate={{ opacity: 1, y: 0 }}
            className="mt-8 rounded-2xl border border-white/10 bg-white/[0.04] p-7 backdrop-blur"
          >
            <div className="mb-6 flex items-center justify-center gap-3">
              <WinIcon size={34} className={STYLE[pred.label].text} />
              <span className={`text-3xl font-extrabold capitalize ${STYLE[pred.label].text}`}>
                {pred.label}
              </span>
              <span className="text-slate-500">
                {Math.round(pred.score * 100)}% confident
              </span>
            </div>
            <div className="space-y-3">
              {ORDER.map((cls) => {
                const v = pred.scores[cls] ?? 0;
                return (
                  <div key={cls} className="flex items-center gap-3">
                    <span className="w-20 text-right text-sm capitalize text-slate-400">
                      {cls}
                    </span>
                    <div className="h-3 flex-1 overflow-hidden rounded-full bg-white/10">
                      <motion.div
                        className={`h-full rounded-full bg-gradient-to-r ${STYLE[cls].bar}`}
                        initial={{ width: 0 }}
                        animate={{ width: `${v * 100}%` }}
                        transition={{ duration: 0.4 }}
                      />
                    </div>
                    <span className="w-12 text-right font-mono text-xs text-slate-500">
                      {(v * 100).toFixed(0)}%
                    </span>
                  </div>
                );
              })}
            </div>
          </motion.div>
        )}

        <footer className="mt-16 text-center text-xs text-slate-600">
          TF-IDF + logistic regression · FastAPI · Docker · real-time inference
        </footer>
      </div>
    </div>
  );
}

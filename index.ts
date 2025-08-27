// src/embed-fetch.ts
import { setTimeout as delay } from "timers/promises";

async function main() {
  const text = process.argv.slice(2).join(" ").trim();
  if (!text) {
    console.error('Usage: npx tsx src/embed-fetch.ts "ma phrase"');
    process.exit(1);
  }

  const BASE = process.env.LMS_BASE_URL ?? "http://localhost:1234/v1";
  const MODEL = process.env.LMS_EMBED_MODEL ?? "text-embedding-qwen3-embedding-4b";

  const resp = await fetch(`${BASE}/embeddings`, {
    method: "POST",
    headers: {
      "Content-Type": "application/json",
      Authorization: `Bearer lm-studio`,
    },
    body: JSON.stringify({ model: MODEL, input: text }),
  });

  if (!resp.ok) {
    console.error("HTTP", resp.status, await resp.text());
    process.exit(1);
  }

  const json = await resp.json();
  const vec: number[] = json.data[0].embedding;
  console.log("Taille du vecteur:", vec.length);
  console.log("Aperçu:", vec.slice(0, 12));
}

main();

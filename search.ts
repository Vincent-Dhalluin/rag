import { readFile } from "fs/promises";
import { join, dirname } from "path";
import { fileURLToPath } from "url";

// Fonction pour calculer la similarité cosinus entre deux vecteurs
function cosineSimilarity(a: number[], b: number[]): number {
  const dot = a.reduce((sum, v, i) => sum + v * b[i], 0);
  const normA = Math.sqrt(a.reduce((sum, v) => sum + v * v, 0));
  const normB = Math.sqrt(b.reduce((sum, v) => sum + v * v, 0));
  return dot / (normA * normB);
}

async function getEmbedding(text: string): Promise<number[]> {
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
  if (!resp.ok) throw new Error(await resp.text());
  const json = await resp.json();
  return json.data[0].embedding;
}

// Interroger le LLM local (lm studio) tinyllama-1.1b-chat-v1.0
async function askLocalLLM(question: string, context: string): Promise<string> {
  const BASE = process.env.LMS_BASE_URL ?? "http://localhost:1234/v1";
  const MODEL = "tinyllama-1.1b-chat-v1.0";
  // Limiter le contexte à 1000 caractères pour TinyLlama
  const shortContext = context.length > 1000 ? context.slice(0, 1000) : context;
  const systemPrompt = `Tu es un assistant qui répond uniquement en utilisant le contexte fourni ci-dessous.\nContexte :\n${shortContext}`;
  const userPrompt = `${question}`;
  const resp = await fetch(`${BASE}/chat/completions`, {
    method: "POST",
    headers: {
      "Content-Type": "application/json",
      Authorization: `Bearer lm-studio`,
    },
    body: JSON.stringify({
      model: MODEL,
      messages: [
        { role: "system", content: systemPrompt },
        { role: "user", content: userPrompt }
      ],
      max_tokens: 512
    }),
  });
  if (!resp.ok) throw new Error(await resp.text());
  const json = await resp.json();
  return json.choices[0].message.content;
}

async function main() {
  const text = process.argv.slice(2).join(" ").trim();
  if (!text) {
    console.error('Usage: npx tsx search.ts "ma phrase"');
    process.exit(1);
  }

  // Récupérer le chemin du fichier vectors_db.txt
  const __filename = fileURLToPath(import.meta.url);
  const __dirname = dirname(__filename);
  const dbPath = join(process.cwd(), "vectors_db.txt");
  const dbRaw = await readFile(dbPath, "utf8");
  const docs = dbRaw
    .split("\n")
    .filter(Boolean)
    .map(line => {
      const parts = line.split("|||");
      if (parts.length < 3) return null; // Ignore les lignes mal formées
      const [file, embeddingStr, textStr] = parts;
      // Utilise une regex pour extraire tous les nombres du vecteur
      const embedding = Array.from(embeddingStr.matchAll(/-?\d+(\.\d+)?/g), m => Number(m[0]));
      const docText = textStr ? JSON.parse(textStr) : "";
      return { file, embedding, docText };
    })
    .filter(Boolean);

  // Générer l'embedding pour le texte saisi
  const inputEmbedding = await getEmbedding(text);

  // Calculer la similarité cosinus et trouver le fichier le plus proche
  let bestDoc: { file: string, embedding: number[], docText: string } | null = null;
  let bestScore = -Infinity;
  for (const doc of docs) {
    const score = cosineSimilarity(inputEmbedding, doc.embedding);
    if (score > bestScore) {
      bestScore = score;
      bestDoc = doc;
    }
  }

  // Utiliser le texte lié au vecteur comme contexte pour le LLM
  if (bestDoc) {
    const answer = await askLocalLLM(text, bestDoc.docText);
    console.log(answer);
  } else {
    console.log("Aucun fichier similaire trouvé.");
  }
}

main();
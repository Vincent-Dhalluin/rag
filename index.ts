// src/embed-fetch.ts
import { readdir, readFile, writeFile } from "fs/promises";
import { join, dirname } from "path";
import { setTimeout as delay } from "timers/promises";
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

async function main() {
  // Récupérer le chemin du dossier documentation
  const __filename = fileURLToPath(import.meta.url);
  const __dirname = dirname(__filename);
  const docDir = join(__dirname, "documentation");
  const files = (await readdir(docDir)).filter(f => f.endsWith(".txt"));

  // Générer les embeddings pour chaque fichier
  const docs: { file: string, embedding: number[] }[] = [];
  for (const file of files) {
    const content = await readFile(join(docDir, file), "utf8");
    const embedding = await getEmbedding(content);
    docs.push({ file, embedding });
    await delay(100); // Pour éviter de spammer l'API
  }

  // Lire tous les contenus des fichiers
  const contents: string[] = [];
  for (const file of files) {
    contents.push(await readFile(join(docDir, file), "utf8"));
  }

  // Enregistrer les vecteurs et le texte dans un fichier txt à la racine
  const dbPath = join(__dirname, "vectors_db.txt");
  const formatEmbedding = (embedding: number[]) =>
    embedding.map(v => `${v},;.\n`).join("");
  const dbContent = docs
    .map((doc, i) => `${doc.file}|||${formatEmbedding(doc.embedding)}|||${JSON.stringify(contents[i])}`)
    .join("\n");
  await writeFile(dbPath, dbContent, "utf8");

  console.log(`Vecteurs et textes enregistrés pour ${docs.length} fichiers.`);
}

main();

import { mkdir, readFile, writeFile } from "node:fs/promises";
import path from "node:path";
import { exportNodeGraph } from "./graphExporter.js";
import { parseProgramDocument, parseRegistryDocument } from "./graphDocuments.js";

async function main() {
  const args = process.argv.slice(2);
  if (args.length === 1 && args[0] === "--help") {
    console.log("Usage: npm run graph -- --program <program.json> --registry <registry.json> --out <graph.json>");
    return;
  }
  const options = new Map<string, string>();
  for (let i = 0; i < args.length; i += 2) {
    const key = args[i];
    const value = args[i + 1];
    if (!["--program", "--registry", "--out"].includes(key) || options.has(key) || !value || value.startsWith("--")) throw new Error(`Invalid or duplicate option ${key}; use --help`);
    options.set(key, value);
  }
  for (const key of ["--program", "--registry", "--out"]) if (!options.has(key)) throw new Error(`Missing ${key}; use --help`);
  const programPath = path.resolve(options.get("--program")!);
  const registryPath = path.resolve(options.get("--registry")!);
  const outputPath = path.resolve(options.get("--out")!);
  const normalize = (p: string) => process.platform === "win32" ? p.toLowerCase() : p;
  if ([programPath, registryPath].some(p => normalize(p) === normalize(outputPath))) throw new Error("Output must not overwrite the program or registry");
  const readJson = async (p: string): Promise<unknown> => {
    try { return JSON.parse(await readFile(p, "utf8")); }
    catch (error) { throw new Error(`${p}: ${error instanceof Error ? error.message : error}`); }
  };
  const program = parseProgramDocument(await readJson(programPath));
  const registry = parseRegistryDocument(await readJson(registryPath));
  const graph = exportNodeGraph(program, registry);
  await mkdir(path.dirname(outputPath), { recursive: true });
  await writeFile(outputPath, JSON.stringify(graph, null, 2) + "\n", "utf8");
  console.log(`Exported ${Object.keys(graph.Nodes).length} nodes and ${Object.keys(graph.Edges).length} edges to ${outputPath}`);
}

main().catch(error => {
  console.error(error instanceof Error ? error.message : error);
  process.exitCode = 1;
});

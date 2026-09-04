import fs from "node:fs/promises";
import path from "node:path";
import { fileURLToPath } from "node:url";
import { SpreadsheetFile, Workbook } from "@oai/artifact-tool";

const root = path.resolve(path.dirname(fileURLToPath(import.meta.url)), "..");
const csvPath = path.join(root, "outputs", "restaurants_labeled.csv");
const qualityPath = path.join(root, "reports", "data_quality.json");
const outputPath = path.join(root, "outputs", "口碑聚类结果.xlsx");
const previewPath = path.join(root, "reports", "workbook_preview.png");

const csvText = await fs.readFile(csvPath, "utf8");
console.log("loaded csv");

function parseCsv(text) {
  const rows = [];
  let row = [];
  let field = "";
  let quoted = false;
  const input = text.charCodeAt(0) === 0xfeff ? text.slice(1) : text;
  for (let i = 0; i < input.length; i += 1) {
    const char = input[i];
    if (quoted) {
      if (char === '"') {
        if (input[i + 1] === '"') {
          field += '"';
          i += 1;
        } else {
          quoted = false;
        }
      } else {
        field += char;
      }
    } else if (char === '"') {
      quoted = true;
    } else if (char === ",") {
      row.push(field);
      field = "";
    } else if (char === "\n") {
      row.push(field.replace(/\r$/, ""));
      rows.push(row);
      row = [];
      field = "";
    } else {
      field += char;
    }
  }
  if (field.length || row.length) {
    row.push(field.replace(/\r$/, ""));
    rows.push(row);
  }
  return rows;
}

const rows = parseCsv(csvText);
const quality = JSON.parse(await fs.readFile(qualityPath, "utf8"));
const expectedRows = Number(quality.restaurant_rows) + 1;
if (rows.length !== expectedRows || rows.some((row) => row.length !== 3)) {
  throw new Error(`CSV shape mismatch: ${rows.length} rows`);
}
console.log(`parsed ${rows.length - 1} data rows`);
const workbook = Workbook.create();
const sheet = workbook.worksheets.add("商家标签");
// 大文件分块写入，避免一次性序列化 24 万行造成长时间阻塞。
const blockSize = 20000;
for (let start = 0; start < rows.length; start += blockSize) {
  const block = rows.slice(start, start + blockSize);
  sheet.getRangeByIndexes(start, 0, block.length, 3).values = block;
  console.log(`wrote ${Math.min(start + block.length, rows.length)}/${rows.length} rows`);
}
console.log("created workbook");
const header = sheet.getRange("A1:C1");
header.format = {
  fill: "#1F4E78",
  font: { name: "Arial", size: 10, bold: true, color: "#FFFFFF" },
  horizontalAlignment: "center",
  verticalAlignment: "center",
};
header.format.rowHeight = 24;
sheet.getRange("A:A").format.columnWidth = 14;
sheet.getRange("B:B").format.columnWidth = 30;
sheet.getRange("C:C").format.columnWidth = 20;
sheet.freezePanes.freezeRows(1);
sheet.showGridLines = false;
console.log("formatted workbook");

const inspect = await workbook.inspect({
  kind: "table",
  range: "商家标签!A1:C20",
  include: "values,formulas",
  tableMaxRows: 20,
  tableMaxCols: 3,
  maxChars: 5000,
});
console.log(inspect.ndjson);
const errors = await workbook.inspect({
  kind: "match",
  searchTerm: "#REF!|#DIV/0!|#VALUE!|#NAME\\?|#N/A|#NUM!|#NULL!|#SPILL!|#CALC!",
  options: { useRegex: true, maxResults: 100 },
  summary: "final formula error scan",
});
console.log(errors.ndjson);

const preview = await workbook.render({
  sheetName: "商家标签",
  range: "A1:C25",
  scale: 1.5,
  format: "png",
});
console.log("rendered preview");
await fs.mkdir(path.dirname(previewPath), { recursive: true });
await fs.writeFile(previewPath, new Uint8Array(await preview.arrayBuffer()));
const xlsx = await SpreadsheetFile.exportXlsx(workbook);
console.log("exported workbook");
await xlsx.save(outputPath);
console.log(`saved ${outputPath}`);

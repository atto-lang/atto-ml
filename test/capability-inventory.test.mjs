import test from 'node:test';
import assert from 'node:assert/strict';
import { readFileSync, existsSync } from 'node:fs';

const root = new URL('../', import.meta.url);
const manifest = JSON.parse(readFileSync(new URL('atto-capabilities.json', root), 'utf8'));

// Baseline inventory only; proposed shared schema: atto-spec#10.
// This does not execute or certify any AttoLang guest.
test('inventory is structurally accurate and makes no execution claims', () => {
  assert.deepEqual(Object.keys(manifest).sort(), ['certification','evidence','feature_status','repository','role','schema','source_state'].sort());
  assert.equal(manifest.schema, 'atto.repository-capabilities.v1');
  assert.equal(manifest.repository, 'atto-lang/atto-ml');
  assert.equal(manifest.role, 'library');
  assert.equal(manifest.source_state, 'implementation');
  assert.ok(existsSync(new URL('src/kan/cubic_bspline_f64.ores', root)));
  assert.deepEqual(manifest.feature_status, {public_api:'not_executed',guest_execution:'not_executed'});
  assert.equal(manifest.certification, 'NOT_CERTIFIED');
  assert.deepEqual(manifest.evidence, []);
});

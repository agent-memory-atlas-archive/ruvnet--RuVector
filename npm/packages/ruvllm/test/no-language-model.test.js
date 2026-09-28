/**
 * The package does not load model files, and the native engine has no
 * language-model weights: say so instead of returning placeholder text silently.
 */
const { test } = require('node:test');
const assert = require('node:assert');
const { spawnSync } = require('node:child_process');
const path = require('node:path');

const pkg = path.join(__dirname, '..');
const run = (body) =>
  spawnSync(process.execPath, ['-e', `const { RuvLLM } = require('./dist/cjs/index.js');\n${body}`], {
    cwd: pkg,
    encoding: 'utf8',
  });
const count = (s, code) => (s.match(new RegExp(code, 'g')) || []).length;

test('modelPath is reported as unsupported, once', () => {
  const r = run("new RuvLLM({ modelPath: './m.gguf' }); new RuvLLM({ modelPath: './m.gguf' });");
  assert.strictEqual(r.status, 0, r.stderr);
  assert.strictEqual(count(r.stderr, 'RUVLLM_UNSUPPORTED_OPTION'), 1);
});

test('strict turns the modelPath warning into an error', () => {
  const r = run("try { new RuvLLM({ modelPath: './m.gguf', strict: true }); console.log('no-throw'); } catch (e) { console.log(e.message.split(':')[0]); }");
  assert.match(r.stdout, /RUVLLM_UNSUPPORTED_OPTION/);
});

test('native generate/query warn once that the text is not model output; strict throws', () => {
  const probe = run('console.log(new RuvLLM().isNativeLoaded())');
  if (probe.stdout.trim() !== 'true') return; // JS fallback already returns an explanation
  const r = run("const l = new RuvLLM(); l.generate('hi', { maxTokens: 4 }); l.query('hi'); l.generate('again');");
  assert.strictEqual(count(r.stderr, 'RUVLLM_NO_LANGUAGE_MODEL'), 1);
  const s = run("try { new RuvLLM({ strict: true }).generate('hi'); console.log('no-throw'); } catch (e) { console.log(e.message.split(':')[0]); }");
  assert.match(s.stdout, /RUVLLM_NO_LANGUAGE_MODEL/);
  const quiet = run("const l = new RuvLLM(); l.embed('hi'); l.route('hi');");
  assert.strictEqual(count(quiet.stderr, 'RUVLLM_NO_LANGUAGE_MODEL'), 0, 'routing/embedding do not warn');
});

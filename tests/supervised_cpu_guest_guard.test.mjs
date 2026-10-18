// Real host-side subprocess tests of the guest verifier's *admission* gate.
// A local synthetic executable is NEVER allowed to run in these cases.
// No Oreslang source, runtime, or guest compiler is executed by this suite.
import test from "node:test";
import assert from "node:assert/strict";
import {spawnSync} from "node:child_process";
import {chmodSync,copyFileSync,existsSync,mkdirSync,mkdtempSync,rmSync,writeFileSync} from "node:fs";
import {tmpdir} from "node:os";
import {join} from "node:path";
import {createHash} from "node:crypto";
import {fileURLToPath} from "node:url";

const verifier=fileURLToPath(new URL("../scripts/verify-supervised-cpu-guest.sh",import.meta.url));
test("guest admission rejects unpinned and invalid executables before any guest invocation",()=>{
  const dir=mkdtempSync(join(tmpdir(),"ores-ml-guest-deny-"));
  try{
    const marker=join(dir,"unexpectedly-executed");
    const binary=join(dir,"fake-oreslang");
    writeFileSync(binary,"#!/usr/bin/env bash\n"+"echo invoked > "+JSON.stringify(marker)+"\n");
    chmodSync(binary,0o700);
    function expectRejected(overrides,reason){
      const env={...process.env,ORESLANG_BIN:binary,ORESLANG_BIN_SHA256:"",...overrides};
      const child=spawnSync("bash",[verifier],{env,cwd:dir,encoding:"utf8",timeout:10000});
      assert.equal(child.status,78,reason+": "+child.stderr);
      assert.match(child.stderr,/NOT_EXECUTED/,reason);
      assert.equal(existsSync(marker),false,reason+" must not run the compiler");
    }
    expectRejected({ORESLANG_BIN:""},"missing executable");
    expectRejected({ORESLANG_BIN:"relative-oreslang"},"nonabsolute executable");
    expectRejected({ORESLANG_BIN_SHA256:""},"missing digest");
    expectRejected({ORESLANG_BIN_SHA256:"f".repeat(63)},"malformed digest");
    expectRejected({ORESLANG_BIN_SHA256:"0".repeat(64)},"digest mismatch");
    chmodSync(binary,0o600);
    expectRejected({ORESLANG_BIN_SHA256:"a".repeat(64)},"not executable");
  }finally{
    rmSync(dir,{recursive:true,force:true});
  }
});

test("correctly hashed fake compiler still cannot run without three approved dependency trees",()=>{
  const dir=mkdtempSync(join(tmpdir(),"ores-ml-no-dependencies-"));
  try{
    const scriptDir=join(dir,"oreslang-ml","scripts");
    mkdirSync(scriptDir,{recursive:true});
    const isolatedVerifier=join(scriptDir,"verify-supervised-cpu-guest.sh");
    copyFileSync(verifier,isolatedVerifier);
    const marker=join(dir,"fake-compiler-ran");
    const executable=join(dir,"synthetic-compiler");
    writeFileSync(executable,"#!/usr/bin/env bash\n"+"echo ran > "+JSON.stringify(marker)+"\n");
    chmodSync(executable,0o700);
    const digest=createHash("sha256").update(
      "#!/usr/bin/env bash\n"+"echo ran > "+JSON.stringify(marker)+"\n"
    ).digest("hex");
    const child=spawnSync("bash",[isolatedVerifier],{
      env:{...process.env,ORESLANG_BIN:executable,ORESLANG_BIN_SHA256:digest},
      cwd:dir,encoding:"utf8",timeout:10000
    });
    assert.equal(child.status,78,child.stderr);
    assert.match(child.stderr,/missing dependency repository oreslang-data-science/);
    assert.equal(existsSync(marker),false,"unapproved dependencies cannot launch a compiler");
  }finally{
    rmSync(dir,{recursive:true,force:true});
  }
});

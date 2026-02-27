// --- Node.js Stdin/Stdout Bridge ---
// Python Gymnasium wrapper communicates with this process via JSON lines.
// Commands: { cmd: "reset", config: {...} }
//           { cmd: "step", action: number | null }
//           { cmd: "close" }
import { resetEnv, stepEnv, getObservation, OBS_SIZE, DEFAULT_CONFIG } from "./headlessEnv.js";
let state = null;
let config = { ...DEFAULT_CONFIG };
function handleCommand(line) {
    let cmd;
    try {
        cmd = JSON.parse(line);
    }
    catch {
        respond({ error: "invalid JSON" });
        return;
    }
    switch (cmd.cmd) {
        case "reset": {
            if (cmd.config) {
                config = { ...DEFAULT_CONFIG, ...cmd.config };
            }
            state = resetEnv(config);
            const obs = getObservation(state);
            respond({ obs, info: { obs_size: OBS_SIZE } });
            break;
        }
        case "step": {
            if (!state) {
                respond({ error: "no active episode — call reset first" });
                return;
            }
            if (state.done) {
                respond({ error: "episode already done — call reset" });
                return;
            }
            const action = cmd.action ?? null;
            const result = stepEnv(state, config, action);
            respond({
                obs: result.obs,
                reward: result.reward,
                done: result.done,
                info: result.info,
            });
            break;
        }
        case "close": {
            respond({ status: "closed" });
            process.exit(0);
            break;
        }
        default:
            respond({ error: `unknown command: ${cmd.cmd}` });
    }
}
function respond(data) {
    process.stdout.write(JSON.stringify(data) + "\n");
}
// --- Line reader from stdin ---
let buffer = "";
process.stdin.setEncoding("utf-8");
process.stdin.on("data", (chunk) => {
    buffer += chunk;
    let newlineIdx;
    while ((newlineIdx = buffer.indexOf("\n")) !== -1) {
        const line = buffer.slice(0, newlineIdx).trim();
        buffer = buffer.slice(newlineIdx + 1);
        if (line.length > 0) {
            handleCommand(line);
        }
    }
});
process.stdin.on("end", () => {
    process.exit(0);
});
// Signal ready
respond({ status: "ready", obs_size: OBS_SIZE });

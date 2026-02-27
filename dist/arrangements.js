// --- Blob Spatial Arrangement Generators ---
// Varied blob layouts to prevent the agent from memorizing spatial shortcuts.
// Each arrangement style produces different counting dynamics.
import { vec2, dist } from "./counting-world/steering.js";
const ALL_ARRANGEMENTS = ["scattered", "clustered", "grid-like", "mixed"];
/** Pick a random arrangement type. */
export function randomArrangement() {
    return ALL_ARRANGEMENTS[Math.floor(Math.random() * ALL_ARRANGEMENTS.length)];
}
/**
 * Generate blob positions for a given arrangement style.
 * Returns an array of Vec2 positions, length === count.
 * May return fewer if placement fails (tight constraints).
 */
export function generateArrangement(count, width, height, margin, minSep, style) {
    switch (style) {
        case "scattered":
            return scatteredPositions(count, width, height, margin, minSep);
        case "clustered":
            return clusteredPositions(count, width, height, margin, minSep);
        case "grid-like":
            return gridLikePositions(count, width, height, margin, minSep);
        case "mixed":
            return mixedPositions(count, width, height, margin, minSep);
        default:
            return scatteredPositions(count, width, height, margin, minSep);
    }
}
/** Uniform random scatter with minimum separation. */
function scatteredPositions(count, w, h, margin, minSep) {
    const positions = [];
    let attempts = 0;
    while (positions.length < count && attempts < count * 80) {
        attempts++;
        const x = margin + Math.random() * (w - 2 * margin);
        const y = margin + Math.random() * (h - 2 * margin);
        const candidate = vec2(x, y);
        if (positions.every((p) => dist(p, candidate) >= minSep)) {
            positions.push(candidate);
        }
    }
    return positions;
}
/** 2-3 tight clusters of blobs. */
function clusteredPositions(count, w, h, margin, minSep) {
    const numClusters = 2 + Math.floor(Math.random() * 2); // 2-3
    const clusterCenters = [];
    // Place cluster centers with good separation
    for (let c = 0; c < numClusters; c++) {
        let attempts = 0;
        while (attempts < 100) {
            attempts++;
            const cx = margin * 2 + Math.random() * (w - 4 * margin);
            const cy = margin * 2 + Math.random() * (h - 4 * margin);
            const candidate = vec2(cx, cy);
            const minClusterDist = Math.min(w, h) * 0.25;
            if (clusterCenters.every((cc) => dist(cc, candidate) >= minClusterDist)) {
                clusterCenters.push(candidate);
                break;
            }
        }
    }
    if (clusterCenters.length === 0) {
        return scatteredPositions(count, w, h, margin, minSep);
    }
    // Distribute blobs among clusters
    const positions = [];
    const clusterRadius = Math.min(w, h) * 0.12;
    let attempts = 0;
    while (positions.length < count && attempts < count * 80) {
        attempts++;
        const center = clusterCenters[Math.floor(Math.random() * clusterCenters.length)];
        const angle = Math.random() * Math.PI * 2;
        const r = Math.random() * clusterRadius;
        const x = center.x + Math.cos(angle) * r;
        const y = center.y + Math.sin(angle) * r;
        if (x < margin || x > w - margin || y < margin || y > h - margin)
            continue;
        const candidate = vec2(x, y);
        if (positions.every((p) => dist(p, candidate) >= minSep)) {
            positions.push(candidate);
        }
    }
    return positions;
}
/** Roughly grid arrangement with noise. */
function gridLikePositions(count, w, h, margin, minSep) {
    const cols = Math.ceil(Math.sqrt(count * (w / h)));
    const rows = Math.ceil(count / cols);
    const cellW = (w - 2 * margin) / cols;
    const cellH = (h - 2 * margin) / rows;
    const jitter = Math.min(cellW, cellH) * 0.25;
    const positions = [];
    for (let r = 0; r < rows && positions.length < count; r++) {
        for (let c = 0; c < cols && positions.length < count; c++) {
            const baseX = margin + (c + 0.5) * cellW;
            const baseY = margin + (r + 0.5) * cellH;
            const x = baseX + (Math.random() - 0.5) * 2 * jitter;
            const y = baseY + (Math.random() - 0.5) * 2 * jitter;
            positions.push(vec2(Math.max(margin, Math.min(w - margin, x)), Math.max(margin, Math.min(h - margin, y))));
        }
    }
    return positions;
}
/** Mix: half clustered, half scattered. */
function mixedPositions(count, w, h, margin, minSep) {
    const clusterCount = Math.floor(count * 0.5);
    const scatterCount = count - clusterCount;
    const clustered = clusteredPositions(clusterCount, w, h, margin, minSep);
    const scattered = [];
    let attempts = 0;
    while (scattered.length < scatterCount && attempts < scatterCount * 80) {
        attempts++;
        const x = margin + Math.random() * (w - 2 * margin);
        const y = margin + Math.random() * (h - 2 * margin);
        const candidate = vec2(x, y);
        const allExisting = [...clustered, ...scattered];
        if (allExisting.every((p) => dist(p, candidate) >= minSep)) {
            scattered.push(candidate);
        }
    }
    return [...clustered, ...scattered];
}

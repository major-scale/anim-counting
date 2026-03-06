// --- Grid-Based Counting ---
// Blobs slide from the field to a visible grid on the right side of the world.
// The grid IS the count — each filled slot represents one counted blob.
// Uncounting reverses in FILO order: last placed → first removed.
import { vec2, dist } from "./steering.js";
import { countBlob, unmarkBlob, startBlobTransition, } from "./blob.js";
// --- Constants ---
export const GRID_COLS = 5;
export const GRID_ROWS = 5;
export const GRID_CELL_SIZE = 50;
// Grid region positioning (right side of 1400×1000 world)
const GRID_RIGHT_MARGIN = 60;
const GRID_WIDTH = GRID_COLS * GRID_CELL_SIZE;
/**
 * Create a 5×5 grid positioned on the right side of the world.
 * Slots fill left-to-right, bottom-to-top (row 1 at bottom).
 */
export function createGrid(worldWidth, worldHeight) {
    const gridLeft = worldWidth - GRID_RIGHT_MARGIN - GRID_WIDTH;
    const gridHeight = GRID_ROWS * GRID_CELL_SIZE;
    const gridTop = (worldHeight - gridHeight) / 2;
    const slots = [];
    // Row 0 = bottom row, row 4 = top row
    for (let row = 0; row < GRID_ROWS; row++) {
        for (let col = 0; col < GRID_COLS; col++) {
            const x = gridLeft + col * GRID_CELL_SIZE + GRID_CELL_SIZE / 2;
            // Bottom-to-top: row 0 has highest y (bottom of screen)
            const y = gridTop + (GRID_ROWS - 1 - row) * GRID_CELL_SIZE + GRID_CELL_SIZE / 2;
            slots.push(vec2(x, y));
        }
    }
    return {
        slots,
        occupancy: new Array(slots.length).fill(-1),
        placementOrder: [],
        filledCount: 0,
        gridLeft,
        gridTop,
    };
}
/**
 * Count a blob into the next available grid slot.
 * Triggers slide animation. Returns false if blob is already gridded.
 */
export function countBlobToGrid(blob, blobIndex, grid, botId, botColor) {
    // Already in the grid
    if (blob.gridSlot !== null)
        return false;
    // Grid full
    if (grid.filledCount >= grid.slots.length)
        return false;
    // Find next empty slot (linear scan — always fills in order)
    const slotIdx = grid.filledCount;
    grid.occupancy[slotIdx] = blobIndex;
    grid.placementOrder.push(blobIndex);
    grid.filledCount++;
    blob.gridSlot = slotIdx;
    countBlob(blob, botId, botColor);
    startBlobTransition(blob, grid.slots[slotIdx]);
    return true;
}
/**
 * Uncount the most recently placed blob (FILO). Slides it back to a scatter position.
 * Returns the blob index and scatter position, or null if grid is empty.
 */
export function uncountBlobFromGrid(field, grid, botId, worldWidth, worldHeight) {
    if (grid.placementOrder.length === 0)
        return null;
    const blobIndex = grid.placementOrder.pop();
    const blob = field.blobs[blobIndex];
    const slotIdx = blob.gridSlot;
    if (slotIdx !== null) {
        grid.occupancy[slotIdx] = -1;
    }
    grid.filledCount--;
    // Clear counting state
    unmarkBlob(blob, botId);
    blob.countFlashColor = "#F87171"; // red flash for uncount
    blob.gridSlot = null;
    // Find a scatter position back in the field
    const scatterPos = findScatterPosition(worldWidth, worldHeight, grid.gridLeft, field.blobs);
    startBlobTransition(blob, scatterPos);
    blob.fieldPosition = { ...scatterPos };
    return { blobIndex, scatterPos };
}
/**
 * Find a random position in the field zone (left side), avoiding other blobs.
 */
export function findScatterPosition(worldWidth, worldHeight, gridLeftEdge, blobs) {
    const margin = 120;
    const fieldMaxX = Math.min(gridLeftEdge - 80, worldWidth * 0.55);
    const minSpacing = 40;
    const maxAttempts = 50;
    for (let i = 0; i < maxAttempts; i++) {
        const x = margin + Math.random() * (fieldMaxX - margin);
        const y = margin + Math.random() * (worldHeight - 2 * margin);
        const candidate = vec2(x, y);
        // Check spacing from blobs that are currently in the field (not in grid)
        const tooClose = blobs.some((b) => {
            if (b.gridSlot !== null)
                return false; // ignore gridded blobs
            return dist(b.position, candidate) < minSpacing;
        });
        if (!tooClose)
            return candidate;
    }
    // Fallback: random position in field zone
    return vec2(margin + Math.random() * (fieldMaxX - margin), margin + Math.random() * (worldHeight - 2 * margin));
}
/**
 * Peek at the next blob to uncount (FILO top) and return its current position.
 * Used for bot navigation during uncount phase.
 */
export function getNextUncountTarget(field, grid) {
    if (grid.placementOrder.length === 0)
        return null;
    const blobIndex = grid.placementOrder[grid.placementOrder.length - 1];
    return field.blobs[blobIndex].position;
}
/**
 * Get the bounding rectangle of the grid region.
 */
export function gridBounds(grid) {
    return {
        left: grid.gridLeft,
        right: grid.gridLeft + GRID_COLS * GRID_CELL_SIZE,
        top: grid.gridTop,
        bottom: grid.gridTop + GRID_ROWS * GRID_CELL_SIZE,
    };
}

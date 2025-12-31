// Game Configuration
const GRID_SIZE = 10; // 🗺️
const CELL_SIZE = 50; // 📏
const PLAYER_HEALTH = 100; // ❤️
const AI_HEALTH = 100; // 🤖❤️
const DAMAGE = 10; // 💥

// Game State 🕹️
let gameState = {
    player: {
        x: 0, // 🗺️
        y: 0, // 🗺️
        health: PLAYER_HEALTH, // ❤️
        direction: 'right' // 👉
    },
    ai: {
        x: 9, // 🗺️
        y: 9, // 🗺️
        health: AI_HEALTH, // ❤️
        direction: 'left' // 👈
    },
    match: {
        number: 1, // 🔢
        playerWins: 0, // 🏆
        aiWins: 0, // 🏆
        moves: [], // 🚶
        currentMatchMoves: 0 // 🚶
    },
    gameActive: false // ❌
};

// AI Timing - ULTRA FAST ✨
let aiMoveInterval = null; // ⏳AI interval ID
let aiNextMoveTime = 0; // ⏱️When AI moves next
let aiIsThinking = false; // 🤔Is AI processing?
let aiReactionTimer = null; // ⏱️AI reaction timer
let aiLastMoveTime = 0; // ⏱️Time AI last moved

// Performance tracking 📊
let aiMoveCount = 0; // ⬆️Total AI moves
let aiTotalReactionTime = 0; // ⏱️Total AI reaction
let playerLastMoveTime = 0; // ⏱️Player last move

// DOM Elements 🖼️
let gameGrid, playerElement, aiElement; // 📍Game elements

// Initialize Game 🚀
function initGame() {
    // Get DOM elements
    gameGrid = document.getElementById('game-grid'); // 🗺️Game grid element
    playerElement = document.getElementById('player'); // 👤Player element
    aiElement = document.getElementById('ai'); // 🤖AI element
    
    // Create grid
    createGrid(); // 🏗️Build the grid
    
    // Position players
    positionPlayer('player', gameState.player.x, gameState.player.y); // 📍Place player start
    positionPlayer('ai', gameState.ai.x, gameState.ai.y); // 📍Place AI start
    
    // Update UI
    updateUI(); // ✍️Refresh game display
    
    // Focus game container
    document.getElementById('game-container').focus(); // 🖱️Game focus input
    
    // Set up keyboard controls
    setupControls(); // ⌨️Player input setup
    
    // Start the game
    startNewMatch(); // ▶️Begin new round
    
    console.log("⚡ ULTRA-FAST AI Game Started!"); // 📣Game start message
    console.log("AI reacts ultra-ultra-fast (10ms)"); // 📣AI speed note
}

// Create 10x10 grid
function createGrid() {
    gameGrid.innerHTML = ''; // Clear old grid 🧹
    for (let y = 0; y < GRID_SIZE; y++) { // Loop rows ⬆️⬇️
        for (let x = 0; x < GRID_SIZE; x++) { // Loop columns ⬅️➡️
            const cell = document.createElement('div'); // Create cell element ✨
            cell.className = 'grid-cell'; // Set class name 🏷️
            cell.dataset.x = x; // Set x-coordinate data 🔢
            cell.dataset.y = y; // Set y-coordinate data 🔢
            gameGrid.appendChild(cell); // Add cell to grid ➕
        }
    }
}

// Position a player on the grid
function positionPlayer(type, x, y) {
    const element = type === 'player' ? playerElement : aiElement; // Get correct element 👤🤖
    element.style.left = `${x * CELL_SIZE + 5}px`; // Set horizontal position ↔️
    element.style.top = `${y * CELL_SIZE + 5}px`; // Set vertical position ↕️
}

// Update UI
function updateUI() {
    // Update health
    document.getElementById('player-health').textContent = gameState.player.health; // Player health text 💖
    document.getElementById('ai-health').textContent = gameState.ai.health; // AI health text 💖
    
    // Update scores
    document.getElementById('match-count').textContent = gameState.match.number; // Match number display 🏆
    document.getElementById('player-wins').textContent = gameState.match.playerWins; // Player wins count 🥳
    document.getElementById('ai-wins').textContent = gameState.match.aiWins; // AI wins count 🤖
    
    // Update health colors
    const playerHealthElement = document.getElementById('player-health'); // Player health element 🔥
    const aiHealthElement = document.getElementById('ai-health'); // AI health element 🔥
    
    playerHealthElement.style.color = gameState.player.health > 50 ? '#00ff00' : 
                                     gameState.player.health > 25 ? '#ffff00' : '#ff0000'; // Player health color 🟢🟡🔴
    
    aiHealthElement.style.color = gameState.ai.health > 50 ? '#ff0000' : 
                                 gameState.ai.health > 25 ? '#ffaa00' : '#ff5500'; // AI health color 🔴🟠
}

// Set up keyboard controls
function setupControls() {
    const gameContainer = document.getElementById('game-container'); // Game container element 📦
    
    gameContainer.addEventListener('keydown', (e) => { // Listen for key presses ⌨️
        if (!gameState.gameActive) return; // Ignore if game inactive 🚫
        
        e.preventDefault(); // Prevent default browser actions ✋
        
        let playerMoved = false; // Track if player moved 🚶
        const moveStartTime = Date.now(); // Record move start time ⏱️
        
        switch(e.key) {
            case 'ArrowUp':
                if (gameState.player.y > 0) { // Check boundary above ⬆️
                    gameState.player.y--; // Move player up ⬆️
                    gameState.player.direction = 'up'; // Set direction up ⬆️
                    playerMoved = true; // Player moved status true ✅
                }
                break;
            case 'ArrowDown':
                if (gameState.player.y < GRID_SIZE - 1) { // Check boundary below ⬇️
                    gameState.player.y++; // Move player down ⬇️
                    gameState.player.direction = 'down'; // Set direction down ⬇️
                    playerMoved = true; // Player moved status true ✅
                }
                break;
            case 'ArrowLeft':
                if (gameState.player.x > 0) { // Check boundary left ⬅️
                    gameState.player.x--; // Move player left ⬅️
                    gameState.player.direction = 'left'; // Set direction left 👈
                    playerMoved = true; // Player moved status true ✅
                }
                break;
            case 'ArrowRight':
                if (gameState.player.x < GRID_SIZE - 1) { // Check boundary right ➡️
                    gameState.player.x++; // Move player right ➡️
                    gameState.player.direction = 'right'; // Set direction right 👉
                    playerMoved = true; // Player moved status true ✅
                }
                break;
            case ' ':
            case 'Spacebar': // older browsers
            case 'Space':    // some browsers use 'Space'
                e.preventDefault(); // Prevent spacebar scroll 🚀
                shoot('player'); // Player shoots projectile 💥
                playerMoved = true; // Player moved status true ✅
                break;
        }
        
        if (playerMoved) {
            // Update player position
            positionPlayer('player', gameState.player.x, gameState.player.y); // Update player visual position 📍
            
            // Record move
            recordMove('player', e.key === ' ' ? 'shoot' : 'move'); // Log the player action 📝
            
            // Update move count
            gameState.match.currentMatchMoves++; // Increment moves for current match 💯
            
            // Record player movement time
            playerLastMoveTime = Date.now(); // Timestamp last player move ⏰
            
            // Trigger INSTANT AI reaction
            triggerAIReaction(); // AI reacts immediately after player move 🧠
            
            // Check win conditions
            checkWinCondition(); // See if game is won or lost 🚩
            
            // Log player move time
            const moveTime = Date.now() - moveStartTime; // Calculate move duration ⏳
            if (moveTime > 16) { // More than one frame at 60fps
                console.log(`Player move processed in ${moveTime}ms`); // Log slow move 🐌
            }
        }
    });
}

// Trigger AI reaction to player movement ⚡🤖
function triggerAIReaction() {
    if (!gameState.gameActive || aiIsThinking) return;
    
    // Calculate time since AI's last move ⏱️
    const timeSinceLastMove = Date.now() - aiLastMoveTime;
    
    // If AI just moved recently, wait a tiny bit (4ms) 🤏
    if (timeSinceLastMove < 4) return;
    
    // Cancel any pending AI move ❌
    if (aiReactionTimer) {
        clearTimeout(aiReactionTimer);
    }
    
    // Schedule AI reaction (fixed 4ms) ⏰
    const reactionDelay = 4; // fixed 4ms
    
    aiReactionTimer = setTimeout(() => {
        aiMove();
    }, reactionDelay);
}

// Start AI thinking cycle 🧠💡
function startAIThinking() {
    if (!gameState.gameActive) return;
    
    // Clear any existing interval 🧹
    if (aiMoveInterval) {
        clearInterval(aiMoveInterval);
    }
    
    // AI thinks at ULTRA-FAST intervals (10ms checks) ⚡⚡
    aiMoveInterval = setInterval(() => {
        if (!gameState.gameActive || aiIsThinking) return;
        
        // Check if it's time for AI to move ⏳
        const now = Date.now();
        if (now >= aiNextMoveTime) {
            aiMove();
        }

        // Even if not time for scheduled move, AI might want to react to player (6ms window) 👁️👂
        const timeSincePlayerMove = now - playerLastMoveTime;
        if (timeSincePlayerMove < 10 && timeSincePlayerMove > 4) {
            // Player moved recently, AI might want to react ⚡
            if (Math.random() < 0.7) {
                triggerAIReaction();
            }
        }
    }, 4); // Check every 4ms (~250 times per second)
}

// Schedule next AI move 🗓️✔️
function scheduleAIMove(delay = null) {
    if (!delay) {
        // Fixed small delay for ultra-ultra-fast AI 💨
        delay = 4; // 4ms (much faster)
    }
    aiNextMoveTime = Date.now() + delay;
}

// AI makes a move 🤖➡️
function aiMove() {
    if (!gameState.gameActive || aiIsThinking) return;
    
    const moveStartTime = Date.now();
    aiIsThinking = true;
    aiMoveCount++;
    
    fetch('/ai_move', {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json',
        },
        body: JSON.stringify({
            player_x: gameState.player.x,
            player_y: gameState.player.y,
            player_health: gameState.player.health,
            ai_x: gameState.ai.x,
            ai_y: gameState.ai.y,
            ai_health: gameState.ai.health,
            match_number: gameState.match.number
        })
    })
    .then(response => response.json())
    .then(data => {
        const processingTime = Date.now() - moveStartTime;
        aiTotalReactionTime += processingTime;
        aiIsThinking = false;
        aiLastMoveTime = Date.now();
        
        if (data.success) {
            if (data.should_wait) {
                // AI decides to wait ⏳
                const waitTime = data.wait_time * 1000;
                console.log(`🤖 AI thinking... (wait ${waitTime.toFixed(0)}ms)`);
                scheduleAIMove(waitTime);
            } else {
                // AI decides to act ✅
                console.log(`⚡ AI: ${data.action} (${data.mode}, ${processingTime}ms)`);
                executeAIAction(data.action);
                
                // Schedule next move promptly (10ms) 🚀
                const nextDelay = 10; // 10ms
                scheduleAIMove(nextDelay);
            }
        }
    })
    .catch(error => {
        console.error('AI move error:', error);
        aiIsThinking = false;
        scheduleAIMove(10); // Retry after 10ms 🔄
    });
}

// Execute AI action 💥🎮
function executeAIAction(action) {
    let aiMoved = false;
    const actionStartTime = Date.now();
    
    switch(action) {
        case 'move_up':
            if (gameState.ai.y > 0) {
                gameState.ai.y--;
                gameState.ai.direction = 'up';
                aiMoved = true;
            }
            break;
        case 'move_down':
            if (gameState.ai.y < GRID_SIZE - 1) {
                gameState.ai.y++;
                gameState.ai.direction = 'down';
                aiMoved = true;
            }
            break;
        case 'move_left':
            if (gameState.ai.x > 0) {
                gameState.ai.x--;
                gameState.ai.direction = 'left';
                aiMoved = true;
            }
            break;
        case 'move_right':
            if (gameState.ai.x < GRID_SIZE - 1) {
                gameState.ai.x++;
                gameState.ai.direction = 'right';
                aiMoved = true;
            }
            break;
        case 'shoot':
            shoot('ai');
            aiMoved = true;
            break;
        case 'wait':
            // AI intentionally waits (for strategic reasons) 🧘
            aiMoved = false;
            break;
    }
    
    if (aiMoved) {
        positionPlayer('ai', gameState.ai.x, gameState.ai.y);
        recordMove('ai', action);
        gameState.match.currentMatchMoves++;
        
        const actionTime = Date.now() - actionStartTime;
        if (actionTime > 10) {
            console.log(`AI action executed in ${actionTime}ms`);
        }
        
        checkWinCondition();
    }
}

// Record a move for training
function recordMove(player, action) {
    const moveData = {
        player: player, // 🧑‍💻 Who acted?
        action: action, // 💥 What happened?
        playerPosition: { x: gameState.player.x, y: gameState.player.y }, // 📍 Player location
        playerHealth: gameState.player.health, // ❤️ Player health
        playerDirection: gameState.player.direction, // 🧭 Player facing
        aiPosition: { x: gameState.ai.x, y: gameState.ai.y }, // 📍 AI location
        aiHealth: gameState.ai.health, // 🤖 AI health
        aiDirection: gameState.ai.direction, // 🧭 AI facing
        matchNumber: gameState.match.number, // 🔢 Current match
        moveNumber: gameState.match.currentMatchMoves, // 🔢 Move index
        timestamp: Date.now() // ⏰ Time of action
    };
    
    gameState.match.moves.push(moveData); // 💾 Store move
}

// Shoot bullet
function shoot(shooter) {
    // Determine shooter & target
    const shooterState = shooter === 'player' ? gameState.player : gameState.ai; // 🎯 Who is shooting?
    const targetState = shooter === 'player' ? gameState.ai : gameState.player; // 🎯 Who is target?

    // Create bullet element
    const bullet = document.createElement('div'); // 💥 New bullet
    bullet.className = `bullet ${shooter}-bullet`; // 🎨 Bullet style

    // Position bullet at shooter (center of cell)
    let startX = shooterState.x * CELL_SIZE + 20; // 📍 Start X pos
    let startY = shooterState.y * CELL_SIZE + 20; // 📍 Start Y pos

    bullet.style.left = `${startX}px`; // ➡️ Bullet X
    bullet.style.top = `${startY}px`; // ⬆️ Bullet Y
    gameGrid.appendChild(bullet); // 🖼️ Add to screen

    // Aim precisely at opponent (fractional direction), but cap range to 2 cells
    const deltaX = (targetState.x - shooterState.x);
    const deltaY = (targetState.y - shooterState.y);
    const distanceCells = Math.sqrt(deltaX * deltaX + deltaY * deltaY);
    const range = 2; // 🧭 Two-block max range

    // If distance is zero (same cell), shoot zero vector
    let dirX = 0, dirY = 0;
    if (distanceCells > 0) {
        dirX = deltaX / distanceCells;
        dirY = deltaY / distanceCells;
    }

    // Determine travel in cells (fractional) and endpoint in cell coords
    const travelCells = Math.min(distanceCells, range);
    const endCellX = shooterState.x + dirX * travelCells;
    const endCellY = shooterState.y + dirY * travelCells;

    // Convert to pixel coordinates (allow fractional cell positions for smooth angle)
    const targetPxX = endCellX * CELL_SIZE + (CELL_SIZE / 2);
    const targetPxY = endCellY * CELL_SIZE + (CELL_SIZE / 2);

    // Animate bullet towards the precise point
    const dx = targetPxX - startX; // ↔️ Horizontal difference
    const dy = targetPxY - startY; // ↕️ Vertical difference
    const distancePx = Math.sqrt(dx * dx + dy * dy); // 📏 Pixel distance

    // Bullet speed (pixels per ms) - tweak for smooth feel
    const bulletSpeed = 1.5; // px/ms (1.5 => 200px in ~133ms)
    const minDuration = 60; // ms
    const maxDuration = 500; // ms
    let duration = Math.round(distancePx / bulletSpeed);
    if (duration < minDuration) duration = minDuration;
    if (duration > maxDuration) duration = maxDuration;

    bullet.animate([
        { transform: 'translate(0, 0) scale(1)', opacity: 1 }, // ▶️ Start state
        { transform: `translate(${dx}px, ${dy}px) scale(0.5)`, opacity: 0 } // ⏹️ End state
    ], {
        duration: duration,
        easing: 'linear'
    });

    // Remove bullet after animation
    setTimeout(() => {
        if (bullet.parentNode) bullet.remove(); // 💨 Bullet gone
    }, duration);

    // Check hit at end of travel (bullet endpoint) — compare actual opponent position
    setTimeout(() => {
        // Calculate opponent center in pixels at check time (use gameState positions)
        const currentTargetPxX = (gameState === undefined ? targetPxX : (gameState[shooter === 'player' ? 'ai' : 'player']?.x ?? targetState.x) * CELL_SIZE + (CELL_SIZE / 2));
        const currentTargetPxY = (gameState === undefined ? targetPxY : (gameState[shooter === 'player' ? 'ai' : 'player']?.y ?? targetState.y) * CELL_SIZE + (CELL_SIZE / 2));

        // Distance between bullet endpoint and current target center
        const endToTargetDx = currentTargetPxX - targetPxX;
        const endToTargetDy = currentTargetPxY - targetPxY;
        const endDist = Math.sqrt(endToTargetDx * endToTargetDx + endToTargetDy * endToTargetDy);

        // If target is within ~60% of a cell radius at endpoint, count as hit
        const hitThreshold = CELL_SIZE * 0.6;
        if (endDist <= hitThreshold) {
            // Apply damage
            targetState.health -= DAMAGE; // 🤕 Target hurt
            if (targetState.health < 0) targetState.health = 0; // 🩸 Don't go below 0
            updateUI(); // 🔄 Refresh display

            // Create hit effect at actual target position
            const hitEffect = document.createElement('div');
            hitEffect.className = 'hit-effect';
            hitEffect.style.left = `${currentTargetPxX - 25}px`;
            hitEffect.style.top = `${currentTargetPxY - 25}px`;
            gameGrid.appendChild(hitEffect);
            setTimeout(() => { if (hitEffect.parentNode) hitEffect.remove(); }, 500);

            if (targetState.health <= 0) {
                endMatch(shooter === 'player' ? 'player' : 'ai');
            }
        }
    }, duration);
}

// Check win condition
function checkWinCondition() {
    if (gameState.player.health <= 0) { // 😵 Player lost
        endMatch('ai'); // 🤖 AI wins
    } else if (gameState.ai.health <= 0) { // 🤖 AI defeated
        endMatch('player'); // 🧑‍💻 Player wins
    }
}

// End match
function endMatch(winner) {
    if (!gameState.gameActive) return; // 🚫 Game over
    if (!winner || winner === 'None') return; // ❓ No winner
    
    gameState.gameActive = false; // 🛑 Game stopped
    
    // Stop AI thinking
    if (aiMoveInterval) { // ✅ AI timer exists
        clearInterval(aiMoveInterval); // 🚫 Stop AI moves
        aiMoveInterval = null; // 🧹 Clean up
    }
    if (aiReactionTimer) { // ✅ AI timer exists
        clearTimeout(aiReactionTimer); // 🚫 Stop AI reaction
        aiReactionTimer = null; // 🧹 Clean up
    }
    
    // Calculate average AI reaction time
    const avgReactionTime = aiMoveCount > 0 ? (aiTotalReactionTime / aiMoveCount) : 0; // 📊 Avg reaction
    
    // Update win count
    if (winner === 'player') { // 🧑‍💻 Player won
        gameState.match.playerWins++; // ✅ Player score up
        console.log(`🎉 PLAYER WINS! AI reacted in avg ${avgReactionTime.toFixed(1)}ms`); // 📣 Announce win
    } else { // 🤖 AI won
        gameState.match.aiWins++; // ✅ AI score up
        console.log(`🤖 AI WINS! Reacted in avg ${avgReactionTime.toFixed(1)}ms`); // 📣 Announce win
    }
    
    // Update UI
    updateUI(); // 🔄 Refresh display
    
    // Send match data to AI
    sendMatchDataToAI(winner); // 📤 Send data
    
    // Start new match after delay
    setTimeout(() => {
        gameState.match.number++; // ⬆️ Next match
        startNewMatch(); // 🚀 New game
    }, 1500); // Short delay between matches ⏳ Wait
}

// Send match data to AI
function sendMatchDataToAI(winner) {
    if (gameState.match.moves.length === 0) return; // No moves yet
    
    console.log(`📤 Sending match ${gameState.match.number} data...`); // Log data sending
    
    fetch('/learn_from_match', {
        method: 'POST', // HTTP POST method
        headers: {
            'Content-Type': 'application/json', // JSON content type
        },
        body: JSON.stringify({
            moves: gameState.match.moves, // Match moves array
            match_number: gameState.match.number, // Current match number
            winner: winner // Match winner identifier
        })
    })
    .then(response => response.json()) // Parse JSON response
    .then(data => {
        if (data.success) {
            console.log(`⚡ AI training complete!`); // AI training success
            console.log(`   Stats: ${data.ai_stats.wins}W ${data.ai_stats.losses}L`); // Display AI stats
            console.log(`   Reaction: ${data.ai_stats.reaction_time_ms?.toFixed(0)}ms`); // Show reaction time
        }
    })
    .catch(error => {
        console.error('Error sending match data:', error); // Log fetch error
    });
}

// Start new match
function startNewMatch() {
    console.log(`\n=== MATCH ${gameState.match.number} STARTING ===`); // Log match start
    
    // Reset performance tracking
    aiMoveCount = 0; // Reset AI move count
    aiTotalReactionTime = 0; // Reset AI reaction total
    playerLastMoveTime = 0; // Reset player last move
    aiLastMoveTime = 0; // Reset AI last move
    
    // Reset positions and health
    gameState.player.x = 0; // Player X position
    gameState.player.y = 0; // Player Y position
    gameState.player.health = PLAYER_HEALTH; // Player starting health
    gameState.player.direction = 'right'; // Player starting direction
    
    gameState.ai.x = 9; // AI X position
    gameState.ai.y = 9; // AI Y position
    gameState.ai.health = AI_HEALTH; // AI starting health
    gameState.ai.direction = 'left'; // AI starting direction
    
    // Reset match moves
    gameState.match.currentMatchMoves = 0; // Reset current match moves
    gameState.match.moves = []; // Clear moves array
    
    // Update positions
    positionPlayer('player', gameState.player.x, gameState.player.y); // Update player position visually
    positionPlayer('ai', gameState.ai.x, gameState.ai.y); // Update AI position visually
    
    // Update UI
    updateUI(); // Refresh game interface
    
    // Start game
    gameState.gameActive = true; // Set game active flag
    
    // Start AI thinking (ULTRA FAST)
    aiNextMoveTime = Date.now() + 4; // AI moves after 4ms
    startAIThinking(); // Initiate AI thought process
    
    // Focus game container
    document.getElementById('game-container').focus(); // Set focus to game area
    
    console.log(`AI will start in 4ms, checking every 4ms!`); // Log AI start timing
}

// Initialize game when page loads
document.addEventListener('DOMContentLoaded', initGame); // Call initGame on load

// Prevent scrolling with arrow keys and space
document.addEventListener('keydown', (e) => {
    if (['ArrowUp', 'ArrowDown', 'ArrowLeft', 'ArrowRight', ' '].includes(e.key)) {
        e.preventDefault(); // Prevent default key action
    }
});

// Performance monitor (optional)
setInterval(() => {
    if (!gameState.gameActive) return; // Only check if game active
    
    const now = Date.now(); // Current timestamp
    const timeSinceAIMove = now - aiLastMoveTime; // Time since AI moved
    
    if (timeSinceAIMove > 20 && !aiIsThinking) {
        // AI hasn't moved in a short time, trigger a move
        triggerAIReaction(); // Force AI to move
    }
}, 100); // Check every 100ms
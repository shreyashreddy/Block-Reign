// Game Configuration
const GRID_SIZE = 10;
const CELL_SIZE = 50;
const PLAYER_HEALTH = 100;
const AI_HEALTH = 100;
const DAMAGE = 25;

// Game State
let gameState = {
    player: {
        x: 0,
        y: 0,
        health: PLAYER_HEALTH,
        direction: 'right'
    },
    ai: {
        x: 9,
        y: 9,
        health: AI_HEALTH,
        direction: 'left'
    },
    match: {
        number: 1,
        playerWins: 0,
        aiWins: 0,
        moves: [],
        currentMatchMoves: 0
    },
    gameActive: false
};

// DOM Elements
let gameGrid, playerElement, aiElement;

// Initialize Game
function initGame() {
    // Get DOM elements
    gameGrid = document.getElementById('game-grid');
    playerElement = document.getElementById('player');
    aiElement = document.getElementById('ai');
    
    // Create grid
    createGrid();
    
    // Position players
    positionPlayer('player', gameState.player.x, gameState.player.y);
    positionPlayer('ai', gameState.ai.x, gameState.ai.y);
    
    // Update UI
    updateUI();
    
    // Focus game container
    document.getElementById('game-container').focus();
    
    // Set up keyboard controls
    setupControls();
    
    // Start the game
    startNewMatch();
    
    console.log("Game started!");
}

// Create 10x10 grid
function createGrid() {
    gameGrid.innerHTML = '';
    for (let y = 0; y < GRID_SIZE; y++) {
        for (let x = 0; x < GRID_SIZE; x++) {
            const cell = document.createElement('div');
            cell.className = 'grid-cell';
            cell.dataset.x = x;
            cell.dataset.y = y;
            gameGrid.appendChild(cell);
        }
    }
}

// Position a player on the grid
function positionPlayer(type, x, y) {
    const element = type === 'player' ? playerElement : aiElement;
    element.style.left = `${x * CELL_SIZE + 5}px`;
    element.style.top = `${y * CELL_SIZE + 5}px`;
}

// Update UI
function updateUI() {
    // Update health
    document.getElementById('player-health').textContent = gameState.player.health;
    document.getElementById('ai-health').textContent = gameState.ai.health;
    
    // Update scores
    document.getElementById('match-count').textContent = gameState.match.number;
    document.getElementById('player-wins').textContent = gameState.match.playerWins;
    document.getElementById('ai-wins').textContent = gameState.match.aiWins;
    
    // Update health colors
    const playerHealthElement = document.getElementById('player-health');
    const aiHealthElement = document.getElementById('ai-health');
    
    playerHealthElement.style.color = gameState.player.health > 50 ? '#00ff00' : 
                                     gameState.player.health > 25 ? '#ffff00' : '#ff0000';
    
    aiHealthElement.style.color = gameState.ai.health > 50 ? '#ff0000' : 
                                 gameState.ai.health > 25 ? '#ffaa00' : '#ff5500';
}

// Set up keyboard controls
function setupControls() {
    const gameContainer = document.getElementById('game-container');
    
    gameContainer.addEventListener('keydown', (e) => {
        if (!gameState.gameActive) return;
        
        e.preventDefault(); // Prevent scrolling
        
        let playerMoved = false;
        
        switch(e.key) {
            case 'ArrowUp':
                if (gameState.player.y > 0) {
                    gameState.player.y--;
                    playerMoved = true;
                }
                break;
            case 'ArrowDown':
                if (gameState.player.y < GRID_SIZE - 1) {
                    gameState.player.y++;
                    playerMoved = true;
                }
                break;
            case 'ArrowLeft':
                if (gameState.player.x > 0) {
                    gameState.player.x--;
                    gameState.player.direction = 'left';
                    playerMoved = true;
                }
                break;
            case 'ArrowRight':
                if (gameState.player.x < GRID_SIZE - 1) {
                    gameState.player.x++;
                    gameState.player.direction = 'right';
                    playerMoved = true;
                }
                break;
            case ' ':
                e.preventDefault();
                shoot('player');
                playerMoved = true;
                break;
        }
        
        if (playerMoved) {
            // Update player position
            positionPlayer('player', gameState.player.x, gameState.player.y);
            
            // Record move
            recordMove('player', e.key === ' ' ? 'shoot' : 'move');
            
            // AI makes a move
            setTimeout(() => aiMove(), 300);
            
            // Update move count
            gameState.match.currentMatchMoves++;
            
            // Check win conditions
            checkWinCondition();
        }
    });
}

// Record a move for training
function recordMove(player, action) {
    const moveData = {
        player: player,
        action: action,
        playerPosition: { x: gameState.player.x, y: gameState.player.y },
        playerHealth: gameState.player.health,
        playerDirection: gameState.player.direction,
        aiPosition: { x: gameState.ai.x, y: gameState.ai.y },
        aiHealth: gameState.ai.health,
        aiDirection: gameState.ai.direction,
        matchNumber: gameState.match.number,
        moveNumber: gameState.match.currentMatchMoves,
        timestamp: Date.now()
    };
    
    gameState.match.moves.push(moveData);
}

// AI makes a move
function aiMove() {
    if (!gameState.gameActive) return;
    
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
        if (data.action) {
            executeAIAction(data.action);
        }
    })
    .catch(error => {
        console.error('AI move error:', error);
        makeRandomAIMove();
    });
}

// Execute AI action
function executeAIAction(action) {
    let aiMoved = false;
    
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
    }
    
    if (aiMoved) {
        positionPlayer('ai', gameState.ai.x, gameState.ai.y);
        recordMove('ai', action);
        gameState.match.currentMatchMoves++;
        checkWinCondition();
    }
}

// Make random AI move (fallback)
function makeRandomAIMove() {
    const actions = ['move_up', 'move_down', 'move_left', 'move_right', 'shoot'];
    const randomAction = actions[Math.floor(Math.random() * actions.length)];
    executeAIAction(randomAction);
}

// Shoot bullet
function shoot(shooter) {
    const shooterState = shooter === 'player' ? gameState.player : gameState.ai;
    const targetState = shooter === 'player' ? gameState.ai : gameState.player;
    
    // Create bullet element
    const bullet = document.createElement('div');
    bullet.className = `bullet ${shooter}-bullet`;
    
    // Position bullet at shooter
    let startX = shooterState.x * CELL_SIZE + 20;
    let startY = shooterState.y * CELL_SIZE + 20;
    
    bullet.style.left = `${startX}px`;
    bullet.style.top = `${startY}px`;
    gameGrid.appendChild(bullet);
    
    // Calculate target position
    let targetX = targetState.x * CELL_SIZE + 20;
    let targetY = targetState.y * CELL_SIZE + 20;
    
    // Calculate direction and distance
    const dx = targetX - startX;
    const dy = targetY - startY;
    const distance = Math.sqrt(dx * dx + dy * dy);
    const duration = Math.min(500, distance);
    
    // Animate bullet
    bullet.animate([
        { transform: 'translate(0, 0) scale(1)', opacity: 1 },
        { transform: `translate(${dx}px, ${dy}px) scale(0.5)`, opacity: 0 }
    ], {
        duration: duration,
        easing: 'linear'
    });
    
    // Remove bullet after animation
    setTimeout(() => {
        if (bullet.parentNode) {
            bullet.remove();
        }
    }, duration);
    
    // Check if hit
    setTimeout(() => {
        const distanceBetween = Math.sqrt(
            Math.pow(shooterState.x - targetState.x, 2) + 
            Math.pow(shooterState.y - targetState.y, 2)
        );
        
        // Hit if within 2 cells
        if (distanceBetween <= 2) {
            targetState.health -= DAMAGE;
            
            // Don't go below 0
            if (targetState.health < 0) targetState.health = 0;
            
            // Update UI
            updateUI();
            
            // Check if target is dead
            if (targetState.health <= 0) {
                endMatch(shooter === 'player' ? 'player' : 'ai');
            }
        }
    }, duration / 2);
}

// Check win condition
function checkWinCondition() {
    if (gameState.player.health <= 0) {
        endMatch('ai');
    } else if (gameState.ai.health <= 0) {
        endMatch('player');
    }
}

// End match
function endMatch(winner) {
    if (!gameState.gameActive) return;
    if (!winner || winner === 'None') return;
    
    gameState.gameActive = false;
    
    // Update win count
    if (winner === 'player') {
        gameState.match.playerWins++;
        console.log(`🎉 PLAYER WINS MATCH ${gameState.match.number}!`);
    } else {
        gameState.match.aiWins++;
        console.log(`🤖 AI WINS MATCH ${gameState.match.number}!`);
    }
    
    // Send match data to AI
    sendMatchDataToAI(winner);
    
    // Start new match after delay
    setTimeout(() => {
        gameState.match.number++;
        startNewMatch();
    }, 2000);
}

// Send match data to AI
function sendMatchDataToAI(winner) {
    if (gameState.match.moves.length === 0) return;
    
    fetch('/learn_from_match', {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json',
        },
        body: JSON.stringify({
            moves: gameState.match.moves,
            match_number: gameState.match.number,
            winner: winner
        })
    })
    .then(response => response.json())
    .then(data => {
        if (data.success) {
            console.log(`✅ AI training complete!`);
        }
    })
    .catch(error => {
        console.error('Error sending match data:', error);
    });
}

// Start new match
function startNewMatch() {
    // Reset positions and health
    gameState.player.x = 0;
    gameState.player.y = 0;
    gameState.player.health = PLAYER_HEALTH;
    gameState.player.direction = 'right';
    
    gameState.ai.x = 9;
    gameState.ai.y = 9;
    gameState.ai.health = AI_HEALTH;
    gameState.ai.direction = 'left';
    
    // Reset match moves
    gameState.match.currentMatchMoves = 0;
    gameState.match.moves = [];
    
    // Update positions
    positionPlayer('player', gameState.player.x, gameState.player.y);
    positionPlayer('ai', gameState.ai.x, gameState.ai.y);
    
    // Update UI
    updateUI();
    
    // Start game
    gameState.gameActive = true;
    
    // Focus game container
    document.getElementById('game-container').focus();
}

// Initialize game when page loads
document.addEventListener('DOMContentLoaded', initGame);

// Prevent scrolling with arrow keys and space
document.addEventListener('keydown', (e) => {
    if (['ArrowUp', 'ArrowDown', 'ArrowLeft', 'ArrowRight', ' '].includes(e.key)) {
        e.preventDefault();
    }
});

"""
Physics Simulator - 2D Particles with Gravity and Elastic Collisions (Pyodide-compatible)

How to run in browser (Pyodide):
- Launch this script with: python physics_simulator.py
- The simulation will prompt for number of sides (0=circle, 3+=polygon) before starting.
- Spacebar: pause/resume. 'R': reset with new random initial conditions.

Note: All interactions are in-browser; no file/network operations.

===========================================================================================
EDUCATIONAL NOTES:
------------------
This Python file implements a real-time, interactive 2D particle physics simulation using:
- Pygame for rendering and event handling
- Numpy for vector arithmetic and performance
- Asyncio for frame timing and Pyodide/browser compatibility

PHYSICS MODELED:
- Each "particle" is a rigid body (either a circle or regular polygon by user choice)
- Particles move under constant gravity (downward acceleration)
- Elastic collisions (conservation of momentum and kinetic energy) between particles, and between particle and walls
- All particles are same size/mass for simplicity

UI INTERACTION:
- User sets particle shape (number of sides) at the start
- Controls: Spacebar to pause/resume; R to reset simulation; H to show/hide help

===========================================================================================
"""

import sys
import random
import math
import asyncio

import pygame
import numpy as np
import platform

# =========================================
# CONSTANTS & SIMULATION PARAMETERS
# =========================================

WIDTH, HEIGHT = 600, 400         # Canvas dimensions in pixels
FPS = 60                         # Frames per second (simulation step rate)
G = 9.8                          # Gravity acceleration (pixels/s^2, applied downward/y+)
RADIUS = 10                      # Radius of bounding circle for each particle (pixels)
MASS = 1                         # Mass per particle (arbitrary units, all equal)
PARTICLE_COUNT = 5               # Number of particles in the simulation

# =========================================
# Platform Detection: Pyodide/Browser vs. Desktop
# =========================================

# Pyodide sets platform.system() == "Emscripten"
ON_BROWSER = platform.system() == "Emscripten"
if not ON_BROWSER:
    print("WARNING: This code is intended for the browser (Pyodide/Emscripten).")


# =========================================
# COLOR UTILITY: Palette Generation
# =========================================

def color_sequence(num):
    """
    Returns a sequence of visually distinct RGB color tuples.

    For full Pyodide compatibility, we avoid advanced HSV conversions and just use a cycling base palette.
    """
    base_palette = [
        (220, 20, 60),     # crimson
        (32, 178, 170),    # light sea green
        (255, 165, 0),     # orange
        (106, 90, 205),    # slate blue
        (50, 205, 50),     # lime green
        (255, 105, 180),   # hot pink
        (255, 215, 0),     # gold
        (255, 69, 0),      # orange red
        (100, 149, 237),   # cornflower blue
        (0, 128, 128),     # teal
    ]
    # Repeat palette if there are more particles than colors defined in base_palette
    colors = []
    while len(colors) < num:
        colors.extend(base_palette)
    return colors[:num]


# =========================================
# PARTICLE CLASS: State, Rendering, Movement
# =========================================

class Particle:
    """
    Represents a 2D particle (circle or regular polygon), moving under gravity, bouncing elastically.
    
    Attributes:
        pos:    [x, y] position (float, center of mass)
        vel:    [vx, vy] velocity (float, pixels/sec)
        mass:   mass of particle (constant)
        radius: bounding radius (collision approximation)
        color:  RGB tuple
        sides:  0=circle, N>=3 => N-sided regular polygon
    """

    def __init__(self, pos, vel, color, sides):
        self.pos = np.array(pos, dtype='float64')     # [x, y] position vector
        self.vel = np.array(vel, dtype='float64')     # [vx, vy] velocity vector
        self.mass = MASS                              # All particles the same mass
        self.radius = RADIUS                          # Collision and rendering radius
        self.color = color                            # For rendering
        self.sides = sides                            # 0 = circle, 3+ = polygon

    def move(self, dt):
        """
        Integrate motion equations for one time-step dt:
            - Gravity applies constant acceleration in +y (downward)
            - Uses basic Newtonian update:
                v(t+dt) = v(t) + a*dt
                r(t+dt) = r(t) + v(t)*dt
        """
        self.vel[1] += G * dt         # Apply gravity (on y component)
        self.pos += self.vel * dt     # Integrate position update

    def draw(self, surf):
        """
        Render the particle on the `surf` Pygame surface.
            - Circle for sides=0
            - Regular N-gon (inscribed in circle) for sides>=3
        """
        if self.sides == 0:
            # Draw as solid circle
            pygame.draw.circle(surf, self.color, tuple(self.pos.astype(int)), self.radius)
        elif self.sides >= 3:
            # Compute vertices of regular N-gon, centered on self.pos
            angle_offset = math.pi / 2  # So first vertex is at top/north
            # Generate each vertex: evenly spaced around a circle ("polygon inscribed in circle")
            vertices = [
                (
                    int(self.pos[0] + self.radius * math.cos(angle_offset + 2 * math.pi * i / self.sides)),
                    int(self.pos[1] - self.radius * math.sin(angle_offset + 2 * math.pi * i / self.sides))
                )
                for i in range(self.sides)
            ]
            pygame.draw.polygon(surf, self.color, vertices)
        # No rendering if sides is invalid

    def reset(self, pos, vel):
        """
        Helper to reset this particle back to a specific position/velocity.
        """
        self.pos[:] = pos
        self.vel[:] = vel


# =========================================
# PHYSICS UTILITIES: Collision Handling
# =========================================

def elastic_collision(p1: Particle, p2: Particle):
    """
    Perform a perfectly elastic collision between two particles (equal mass).

    Updates their velocities based on momentum and energy conservation:
        - Approximates all shapes with their bounding circles for collision purposes
        - Standard vector elastic collision for 2 particles of equal mass

    Formula Reference:
        See https://en.wikipedia.org/wiki/Elastic_collision#Two-dimensional

    Math:
        Let n = (p1.pos - p2.pos)  # vector from p2 to p1
        Let v_rel = p1.vel - p2.vel
        Velocity update (equal mass, 2D):
            p1.vel -= (dot(v_rel, n) / |n|^2) * n
            p2.vel += (dot(v_rel, n) / |n|^2) * n

    Notes:
        - If impact_factor > 0, particles are moving away, so skip updating (no collision on separation)
        - If centers overlap, small random n is chosen to avoid div by zero
    """
    # Vector difference between centers
    delta_pos = p1.pos - p2.pos
    dist_sqr = np.dot(delta_pos, delta_pos)
    if dist_sqr == 0:
        # Overlapping: randomize separation direction to avoid numerical instability
        delta_pos = np.random.rand(2) - 0.5
        dist_sqr = np.dot(delta_pos, delta_pos)
    delta_vel = p1.vel - p2.vel
    impact_factor = np.dot(delta_vel, delta_pos) / dist_sqr
    if impact_factor > 0:
        # Particles moving away; no collision
        return

    # Update velocities using conservation of momentum + energy
    p1.vel -= impact_factor * delta_pos
    p2.vel += impact_factor * delta_pos

def check_particle_boundary(p: Particle):
    """
    Handles elastic collision of particle with simulation boundaries (canvas walls).
        - If a coordinate exceeds left/right or top/bottom, reflect it
        - Sets position to be just inside boundary, reverses velocity component
    """
    for i, limit in enumerate((WIDTH, HEIGHT)):  # i=0 (x-axis), i=1 (y-axis)
        if p.pos[i] - p.radius < 0:
            # Left or Top wall
            p.pos[i] = p.radius
            p.vel[i] = abs(p.vel[i])  # Ensure particle bounces inward
        elif p.pos[i] + p.radius > limit:
            # Right or Bottom wall
            p.pos[i] = limit - p.radius
            p.vel[i] = -abs(p.vel[i]) # Reflect velocity back into area


# =========================================
# USER INPUT: Prompt for Sides (Startup)
# =========================================

def prompt_sides():
    """
    Show a simple prompt for the user to pick polygon sides (0 for circle, 3+ for regular polygon).

    Achieved using a mini Pygame input screen (since Pyodide doesn't allow standard input()).

    Returns:
        sides (int): 0 (circle) or >=3 (polygon)
    """
    pygame.font.init()
    font = pygame.font.SysFont(None, 36)
    info = "Sides per particle (0=circle, 3+=polygon):"
    text = ""
    clock = pygame.time.Clock()
    # Start temporary small window
    screen = pygame.display.set_mode((420, 140))
    pygame.display.set_caption("Particle Shape Input")
    active = True
    while active:
        # Screen redraw:
        screen.fill((245,245,245))
        # Show instructions
        label = font.render(info, True, (0, 0, 0))
        screen.blit(label, (20, 28))
        # Show typing (or underscore if empty)
        user_inp = font.render(text or "_", True, (64, 120, 220))
        screen.blit(user_inp, (20, 72))
        pygame.display.flip()
        # Handle events (quit, keys)
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit(); sys.exit()
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_RETURN:
                    try:
                        val = int(text)
                        if val == 0 or val >= 3:
                            active = False
                            break
                    except Exception:
                        pass  # Not a valid int
                    # Flash invalid input
                    info = "Enter 0 or 3+"
                elif event.key == pygame.K_BACKSPACE:
                    text = text[:-1]
                elif event.key <= 127:
                    ch = event.unicode
                    if ch.isdigit(): text += ch
        clock.tick(30)
    pygame.display.quit()
    return val


# =========================================
# MAIN LOOP: Physics System, Event Loop, Rendering
# =========================================

async def main():
    """
    Orchestrates full simulation startup, main event/render loop, and user input.

    - Prompts for number of sides
    - Sets up simulation window & particles
    - Core event/game loop cycles: input, physics update, rendering
    - Uses asyncio.sleep for timing (Pyodide/browser-friendly)

    Controls:
        Spacebar  - Pause/Resume simulation
        R         - Reset to new random initial
        H         - Show/hide help overlay

    All physics and UI are real-time, with consistent step interval (1/FPS seconds).
    """

    # ---- SHAPE SELECTION ----
    pygame.init()
    sides = prompt_sides()          # Ask user "circle (0) / polygon (3+)" at start
    pygame.display.quit()
    # ---- MAIN SIMULATION WINDOW ----
    pygame.init()
    screen = pygame.display.set_mode((WIDTH, HEIGHT))
    pygame.display.set_caption("Physics Simulator - Lovable, Pyodide-ready")

    clock = pygame.time.Clock()
    font = pygame.font.SysFont(None, 24)

    # ---- PARTICLE STATE: INITIALIZATION ----
    def random_state():
        """
        Generates list of Particle objects with random initial positions/velocities.
        Ensures no two particles overlap at spawn.
        """
        particles = []
        colors = color_sequence(PARTICLE_COUNT)
        for i in range(PARTICLE_COUNT):
            while True:
                pos = np.array([
                    random.uniform(RADIUS+1, WIDTH-RADIUS-1),
                    random.uniform(RADIUS+1, HEIGHT-RADIUS-1)
                ])
                # Check spawn not overlapping with previous
                if all(np.linalg.norm(pos-p.pos) > 2*RADIUS for p in particles):
                    break
            vel = np.array([
                random.uniform(-50, 50),   # vx in pixels/sec
                random.uniform(-50, 50)    # vy in pixels/sec
            ])
            # Each particle gets unique color and same polygon spec
            particles.append(Particle(pos, vel, colors[i], sides))
        return particles

    initial_particles = random_state()

    running = True
    paused = False
    # Deepcopy state for reset
    particles = [Particle(np.copy(p.pos), np.copy(p.vel), p.color, p.sides) for p in initial_particles]
    show_help = True

    while running:
        # ---- EVENT HANDLING ----
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit(); return
            elif event.type == pygame.KEYDOWN:
                # Space: Pause/resume
                if event.key == pygame.K_SPACE:
                    paused = not paused
                # R: Reset simulation
                elif event.key in (pygame.K_r, pygame.K_R):
                    initial_particles = random_state()
                    particles = [Particle(np.copy(p.pos), np.copy(p.vel), p.color, p.sides) for p in initial_particles]
                    paused = False
                # H: Toggle help
                elif event.key in (pygame.K_h, pygame.K_H):
                    show_help = not show_help

        # ---- PHYSICS SIMULATION (if running) ----
        if not paused:
            dt = 1.0 / FPS
            # Move: integrate equations of motion under gravity
            for p in particles:
                p.move(dt)

            # Handle collisions with walls
            for p in particles:
                check_particle_boundary(p)

            # Handle collisions between particles
            for i in range(PARTICLE_COUNT):
                for j in range(i+1, PARTICLE_COUNT):
                    p1, p2 = particles[i], particles[j]
                    # Approximate all shapes as circles for collision math
                    delta = p1.pos - p2.pos
                    dist = np.linalg.norm(delta)
                    if dist < p1.radius + p2.radius:
                        # Overlap detected: push apart slightly to prevent sticking
                        overlap = (p1.radius + p2.radius) - dist + 0.1
                        direction = delta / (dist if dist!=0 else 1)
                        p1.pos += direction * (overlap/2)
                        p2.pos -= direction * (overlap/2)
                        elastic_collision(p1, p2)

        # ---- RENDERING ----
        screen.fill((250,250,255))            # Soft blue background
        for p in particles:
            p.draw(screen)                    # Draw all particles

        # ---- INFO OVERLAYS / INSTRUCTIONS ----
        overlay = f"{'Paused' if paused else 'Running'}   (Space: pause/resume, R: reset, H: help)"
        ovtxt = font.render(overlay, True, (90,90,90))
        screen.blit(ovtxt, (10, HEIGHT - 32))

        if show_help:
            info_lines = [
                "2D Physics Simulator: 5 particles (equal mass/size)",
                f"Shape: {'Circle' if sides==0 else f'{sides}-gon'}",
                "Physics: Gravity, Elastic Collisions (circles/polygons)",
                "Controls:",
                "  Space: pause/resume   R: reset simulation   H: hide/show help",
            ]
            for idx, line in enumerate(info_lines):
                help_txt = font.render(line, True, (50,50,90))
                screen.blit(help_txt, (15, 20+20*idx))

        pygame.display.flip()

        # ---- TIMING: CONTROL FRAME RATE ----
        # asyncio.sleep is required for Pyodide compatibility
        await asyncio.sleep(1.0/FPS)


# =========================================
# PYODIDE / BROWSER ENTRYPOINT
# =========================================

# Use the standard idiom to support both:
#   - Pyodide/browser (platform.system() == "Emscripten")
#   - Desktop interpreter (platform.system() != "Emscripten")
if ON_BROWSER:
    import asyncio
    asyncio.ensure_future(main())  # Pyodide wants "fire and forget" coroutine
else:
    # Run as normal script using asyncio
    asyncio.run(main())


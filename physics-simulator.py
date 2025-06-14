"""
Physics Simulator - 2D Particles with Gravity, Elastic Collisions, and Interactive Cursor Controls (Pyodide-compatible)

How to run in browser (Pyodide):
- Launch this script with: python physics_simulator.py
- The simulation will prompt for number of sides (0=circle, 3+=polygon) before starting.
- Spacebar: pause/resume. 'R': reset with new random initial conditions.
- Interactive controls: Click to create particles, drag to move them, hold keys for special modes

Note: All interactions are in-browser; no file/network operations.

===========================================================================================
EDUCATIONAL NOTES:
------------------
This Python file implements a real-time, interactive 2D particle physics simulation using:
- Pygame for rendering and event handling
- Numpy for vector arithmetic and performance
- Asyncio for frame timing and Pyodide/browser compatibility
- Interactive cursor controls for real-time particle manipulation

PHYSICS MODELED:
- Each "particle" is a rigid body (either a circle or regular polygon by user choice)
- Particles move under constant gravity (downward acceleration)
- Elastic collisions (conservation of momentum and kinetic energy) between particles, and between particle and walls
- Interactive cursor forces (gravity wells, repulsion fields)
- All particles are same size/mass for simplicity (configurable via mouse wheel)

UI INTERACTION:
- User sets particle shape (number of sides) at the start
- Controls: Spacebar to pause/resume; R to reset simulation; H to show/hide help
- Mouse: Left-click to create particles, right-click drag for velocity, wheel for size
- Special modes: G+mouse for gravity well, R+mouse for repulsion, Shift+drag to throw particles

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
RADIUS = 10                      # Default radius of bounding circle for each particle (pixels)
MASS = 1                         # Mass per particle (arbitrary units, all equal)
PARTICLE_COUNT = 5               # Initial number of particles in the simulation
MAX_PARTICLES = 50               # Maximum allowed particles

# Cursor interaction constants
CURSOR_FORCE_RADIUS = 100        # Radius of cursor force effects
CURSOR_FORCE_STRENGTH = 500      # Strength of cursor forces
VELOCITY_SCALE = 0.5             # Scale factor for velocity when dragging particles

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
        selected: boolean indicating if particle is selected
        trail: list of recent positions for trail effect
    """

    def __init__(self, pos, vel, color, sides, radius=RADIUS):
        self.pos = np.array(pos, dtype='float64')     # [x, y] position vector
        self.vel = np.array(vel, dtype='float64')     # [vx, vy] velocity vector
        self.mass = MASS                              # All particles the same mass
        self.radius = radius                          # Collision and rendering radius
        self.color = color                            # For rendering
        self.sides = sides                            # 0 = circle, 3+ = polygon
        self.selected = False                         # Selection state for interactions
        self.trail = []                               # Trail of recent positions
        self.max_trail_length = 20                    # Maximum trail length

    def move(self, dt):
        """
        Integrate motion equations for one time-step dt:
            - Gravity applies constant acceleration in +y (downward)
            - Uses basic Newtonian update:
                v(t+dt) = v(t) + a*dt
                r(t+dt) = r(t) + v(t)*dt
        """
        # Add current position to trail
        self.trail.append(tuple(self.pos))
        if len(self.trail) > self.max_trail_length:
            self.trail.pop(0)
        
        self.vel[1] += G * dt         # Apply gravity (on y component)
        self.pos += self.vel * dt     # Integrate position update

    def apply_cursor_force(self, cursor_pos, force_type, strength, dt):
        """
        Apply cursor-based forces (gravity well or repulsion)
        
        Args:
            cursor_pos: (x, y) position of cursor
            force_type: 'attract' or 'repel'
            strength: force magnitude
            dt: time step
        """
        cursor_vec = np.array(cursor_pos)
        delta = cursor_vec - self.pos
        distance = np.linalg.norm(delta)
        
        if distance < CURSOR_FORCE_RADIUS and distance > 0:
            # Force decreases with distance squared
            force_magnitude = strength / (distance * distance + 1)
            force_direction = delta / distance
            
            if force_type == 'attract':
                force = force_direction * force_magnitude
            else:  # repel
                force = -force_direction * force_magnitude
            
            # Apply force (F = ma, so a = F/m)
            acceleration = force / self.mass
            self.vel += acceleration * dt

    def contains_point(self, point):
        """Check if a point is inside this particle"""
        distance = np.linalg.norm(np.array(point) - self.pos)
        return distance <= self.radius

    def draw(self, surf):
        """
        Render the particle on the `surf` Pygame surface.
            - Circle for sides=0
            - Regular N-gon (inscribed in circle) for sides>=3
            - Draw trail if available
            - Highlight if selected
        """
        # Draw trail
        if len(self.trail) > 1:
            for i in range(1, len(self.trail)):
                alpha = i / len(self.trail)  # Fade effect
                trail_color = tuple(int(c * alpha * 0.3) for c in self.color)
                if i < len(self.trail):
                    pygame.draw.line(surf, trail_color, self.trail[i-1], self.trail[i], 2)
        
        # Draw selection highlight
        if self.selected:
            pygame.draw.circle(surf, (255, 255, 255), tuple(self.pos.astype(int)), self.radius + 3, 2)
        
        # Draw particle
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

    def reset(self, pos, vel):
        """
        Helper to reset this particle back to a specific position/velocity.
        """
        self.pos[:] = pos
        self.vel[:] = vel
        self.trail = []
        self.selected = False


# =========================================
# CURSOR INTERACTION SYSTEM
# =========================================

class CursorSystem:
    """
    Handles all cursor-based interactions with the simulation
    """
    
    def __init__(self):
        self.mouse_pos = (0, 0)
        self.dragging_particle = None
        self.drag_start_pos = None
        self.drag_start_vel = None
        self.creating_velocity = False
        self.velocity_start = None
        self.current_particle_radius = RADIUS
        self.force_mode = None  # 'gravity', 'repulsion', or None
        
    def update_mouse_pos(self, pos):
        """Update current mouse position"""
        self.mouse_pos = pos
    
    def handle_mouse_down(self, pos, button, particles, sides):
        """
        Handle mouse button press events
        
        Args:
            pos: mouse position (x, y)
            button: pygame mouse button constant
            particles: list of particles
            sides: particle shape specification
        """
        if button == 1:  # Left click
            # Check if clicking on existing particle
            clicked_particle = None
            for particle in particles:
                if particle.contains_point(pos):
                    clicked_particle = particle
                    break
            
            if clicked_particle:
                # Start dragging existing particle
                self.dragging_particle = clicked_particle
                self.drag_start_pos = np.copy(clicked_particle.pos)
                self.drag_start_vel = np.copy(clicked_particle.vel)
                clicked_particle.selected = True
            else:
                # Create new particle at click position
                if len(particles) < MAX_PARTICLES:
                    colors = color_sequence(len(particles) + 1)
                    new_color = colors[len(particles)]
                    new_particle = Particle(pos, [0, 0], new_color, sides, self.current_particle_radius)
                    particles.append(new_particle)
        
        elif button == 3:  # Right click
            # Start velocity creation mode
            self.creating_velocity = True
            self.velocity_start = pos
    
    def handle_mouse_up(self, pos, button, particles):
        """Handle mouse button release events"""
        if button == 1:  # Left click release
            if self.dragging_particle:
                # Apply throw velocity if shift was held during drag
                keys = pygame.key.get_pressed()
                if keys[pygame.K_LSHIFT] or keys[pygame.K_RSHIFT]:
                    drag_delta = np.array(pos) - self.drag_start_pos
                    throw_velocity = drag_delta * VELOCITY_SCALE
                    self.dragging_particle.vel = throw_velocity
                
                self.dragging_particle.selected = False
                self.dragging_particle = None
                self.drag_start_pos = None
                self.drag_start_vel = None
        
        elif button == 3:  # Right click release
            if self.creating_velocity and self.velocity_start:
                # Find particle to apply velocity to
                target_particle = None
                for particle in particles:
                    if particle.contains_point(self.velocity_start):
                        target_particle = particle
                        break
                
                if target_particle:
                    # Calculate and apply velocity
                    velocity_delta = np.array(pos) - np.array(self.velocity_start)
                    target_particle.vel = velocity_delta * VELOCITY_SCALE
                
                self.creating_velocity = False
                self.velocity_start = None
    
    def handle_mouse_motion(self, pos, particles):
        """Handle mouse movement"""
        self.mouse_pos = pos
        
        if self.dragging_particle:
            # Move dragged particle to mouse position
            self.dragging_particle.pos = np.array(pos, dtype='float64')
    
    def handle_mouse_wheel(self, direction):
        """Handle mouse wheel for particle size adjustment"""
        if direction > 0:
            self.current_particle_radius = min(self.current_particle_radius + 2, 30)
        else:
            self.current_particle_radius = max(self.current_particle_radius - 2, 5)
    
    def update_force_mode(self, keys):
        """Update force mode based on key presses"""
        if keys[pygame.K_g]:
            self.force_mode = 'gravity'
        elif keys[pygame.K_r]:
            self.force_mode = 'repulsion'
        else:
            self.force_mode = None
    
    def apply_cursor_forces(self, particles, dt):
        """Apply cursor-based forces to particles"""
        if self.force_mode:
            for particle in particles:
                if self.force_mode == 'gravity':
                    particle.apply_cursor_force(self.mouse_pos, 'attract', CURSOR_FORCE_STRENGTH, dt)
                elif self.force_mode == 'repulsion':
                    particle.apply_cursor_force(self.mouse_pos, 'repel', CURSOR_FORCE_STRENGTH, dt)
    
    def draw_cursor_effects(self, surf):
        """Draw visual effects around cursor"""
        # Draw crosshairs
        pygame.draw.line(surf, (255, 255, 255), 
                        (self.mouse_pos[0] - 10, self.mouse_pos[1]), 
                        (self.mouse_pos[0] + 10, self.mouse_pos[1]), 2)
        pygame.draw.line(surf, (255, 255, 255), 
                        (self.mouse_pos[0], self.mouse_pos[1] - 10), 
                        (self.mouse_pos[0], self.mouse_pos[1] + 10), 2)
        
        # Draw force field effects
        if self.force_mode == 'gravity':
            pygame.draw.circle(surf, (0, 255, 0), self.mouse_pos, CURSOR_FORCE_RADIUS, 2)
            # Draw attraction arrows (simplified)
            for angle in range(0, 360, 45):
                rad = math.radians(angle)
                start_x = self.mouse_pos[0] + math.cos(rad) * (CURSOR_FORCE_RADIUS - 20)
                start_y = self.mouse_pos[1] + math.sin(rad) * (CURSOR_FORCE_RADIUS - 20)
                end_x = self.mouse_pos[0] + math.cos(rad) * (CURSOR_FORCE_RADIUS - 10)
                end_y = self.mouse_pos[1] + math.sin(rad) * (CURSOR_FORCE_RADIUS - 10)
                pygame.draw.line(surf, (0, 255, 0), (start_x, start_y), (end_x, end_y), 2)
        
        elif self.force_mode == 'repulsion':
            pygame.draw.circle(surf, (255, 0, 0), self.mouse_pos, CURSOR_FORCE_RADIUS, 2)
            # Draw repulsion arrows
            for angle in range(0, 360, 45):
                rad = math.radians(angle)
                start_x = self.mouse_pos[0] + math.cos(rad) * (CURSOR_FORCE_RADIUS - 30)
                start_y = self.mouse_pos[1] + math.sin(rad) * (CURSOR_FORCE_RADIUS - 30)
                end_x = self.mouse_pos[0] + math.cos(rad) * (CURSOR_FORCE_RADIUS - 10)
                end_y = self.mouse_pos[1] + math.sin(rad) * (CURSOR_FORCE_RADIUS - 10)
                pygame.draw.line(surf, (255, 0, 0), (end_x, end_y), (start_x, start_y), 2)
        
        # Draw velocity creation line
        if self.creating_velocity and self.velocity_start:
            pygame.draw.line(surf, (255, 255, 0), self.velocity_start, self.mouse_pos, 3)
            # Draw arrow head
            angle = math.atan2(self.mouse_pos[1] - self.velocity_start[1], 
                             self.mouse_pos[0] - self.velocity_start[0])
            arrow_length = 15
            arrow_angle = 0.5
            
            arrow_x1 = self.mouse_pos[0] - arrow_length * math.cos(angle - arrow_angle)
            arrow_y1 = self.mouse_pos[1] - arrow_length * math.sin(angle - arrow_angle)
            arrow_x2 = self.mouse_pos[0] - arrow_length * math.cos(angle + arrow_angle)
            arrow_y2 = self.mouse_pos[1] - arrow_length * math.sin(angle + arrow_angle)
            
            pygame.draw.line(surf, (255, 255, 0), self.mouse_pos, (arrow_x1, arrow_y1), 3)
            pygame.draw.line(surf, (255, 255, 0), self.mouse_pos, (arrow_x2, arrow_y2), 3)


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
    - Handles interactive cursor controls

    Controls:
        Spacebar     - Pause/Resume simulation
        R            - Reset to new random initial
        H            - Show/hide help overlay
        G + mouse    - Create gravity well at cursor
        R + mouse    - Create repulsion field at cursor
        Left click   - Create particle or drag existing particle
        Right drag   - Set particle velocity
        Mouse wheel  - Adjust particle size for new particles
        Shift + drag - Throw particle with velocity

    All physics and UI are real-time, with consistent step interval (1/FPS seconds).
    """

    # ---- SHAPE SELECTION ----
    pygame.init()
    sides = prompt_sides()          # Ask user "circle (0) / polygon (3+)" at start
    pygame.display.quit()
    # ---- MAIN SIMULATION WINDOW ----
    pygame.init()
    screen = pygame.display.set_mode((WIDTH, HEIGHT))
    pygame.display.set_caption("Interactive Physics Simulator - Lovable, Pyodide-ready")

    clock = pygame.time.Clock()
    font = pygame.font.SysFont(None, 24)
    small_font = pygame.font.SysFont(None, 18)

    # Initialize cursor system
    cursor_system = CursorSystem()

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
    particles = [Particle(np.copy(p.pos), np.copy(p.vel), p.color, p.sides, p.radius) for p in initial_particles]
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
                    if not pygame.key.get_pressed()[pygame.K_r]:  # Only if R key for reset, not repulsion
                        initial_particles = random_state()
                        particles = [Particle(np.copy(p.pos), np.copy(p.vel), p.color, p.sides, p.radius) for p in initial_particles]
                        paused = False
                # H: Toggle help
                elif event.key in (pygame.K_h, pygame.K_H):
                    show_help = not show_help
            
            elif event.type == pygame.MOUSEBUTTONDOWN:
                cursor_system.handle_mouse_down(event.pos, event.button, particles, sides)
            
            elif event.type == pygame.MOUSEBUTTONUP:
                cursor_system.handle_mouse_up(event.pos, event.button, particles)
            
            elif event.type == pygame.MOUSEMOTION:
                cursor_system.handle_mouse_motion(event.pos, particles)
            
            elif event.type == pygame.MOUSEWHEEL:
                cursor_system.handle_mouse_wheel(event.y)

        # Update cursor system with current key states
        keys = pygame.key.get_pressed()
        cursor_system.update_force_mode(keys)

        # ---- PHYSICS SIMULATION (if running) ----
        if not paused:
            dt = 1.0 / FPS
            
            # Apply cursor forces
            cursor_system.apply_cursor_forces(particles, dt)
            
            # Move: integrate equations of motion under gravity
            for p in particles:
                if p != cursor_system.dragging_particle:  # Don't move dragged particles
                    p.move(dt)

            # Handle collisions with walls
            for p in particles:
                check_particle_boundary(p)

            # Handle collisions between particles
            for i in range(len(particles)):
                for j in range(i+1, len(particles)):
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
        screen.fill((20, 20, 40))            # Dark blue background
        
        # Draw particles
        for p in particles:
            p.draw(screen)
        
        # Draw cursor effects
        cursor_system.draw_cursor_effects(screen)

        # ---- INFO OVERLAYS / INSTRUCTIONS ----
        status_text = f"{'Paused' if paused else 'Running'} | Particles: {len(particles)}/{MAX_PARTICLES} | Radius: {cursor_system.current_particle_radius}"
        if cursor_system.force_mode:
            status_text += f" | Mode: {cursor_system.force_mode.upper()}"
        
        status_surface = font.render(status_text, True, (200, 200, 200))
        screen.blit(status_surface, (10, HEIGHT - 32))
        
        # Mouse coordinates
        coord_text = f"Mouse: ({cursor_system.mouse_pos[0]}, {cursor_system.mouse_pos[1]})"
        coord_surface = small_font.render(coord_text, True, (150, 150, 150))
        screen.blit(coord_surface, (10, HEIGHT - 50))

        if show_help:
            help_lines = [
                "Interactive 2D Physics Simulator",
                f"Shape: {'Circle' if sides==0 else f'{sides}-gon'} | Physics: Gravity + Elastic Collisions",
                "Controls:",
                "  Space: pause/resume   R: reset   H: hide/show help",
                "  Left click: create particle or drag to move",
                "  Right drag: set velocity   Wheel: adjust size",
                "  G + mouse: gravity well   R + mouse: repulsion",
                "  Shift + drag: throw particle with velocity",
            ]
            for idx, line in enumerate(help_lines):
                color = (100, 150, 255) if idx < 2 else (180, 180, 180)
                help_txt = small_font.render(line, True, color)
                screen.blit(help_txt, (15, 15 + 16*idx))

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

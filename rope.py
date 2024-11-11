from cmu_graphics import *
import numpy as np

# constants
G = 1000
JAKOBSEN_ITERATIONS = 20
FPS = 100
DT = 1/FPS

# creates vectors
def vec(x, y):
    return np.array([x, y])

def distance(x1, x2):
    r = x2 - x1
    return np.dot(r, r)**0.5

class Particle:

    def __init__(self, x, r, m):
        self.m = m
        self.r = r
        self.x = x
        self.xPrev = x
        self.f = vec(0, 0)
        self.fixed = False

    # https://en.wikipedia.org/wiki/Verlet_integration
    def update(self):
        a = self.f / self.m
        temp = self.x 
        self.x = (2 * self.x) - self.xPrev + (a * DT**2)
        self.xPrev = temp

        self.v = (self.x - self.xPrev) / DT
        self.f = vec(0, 0)

    def add_gravity(self):
        gVec = self.m * G * vec(0, 1)
        self.f += gVec

    def draw(self):
        x, y = float(self.x[0]), float(self.x[1])
        drawCircle(x, y, self.r)

class Rope:

    def __init__(self, start, end, num_particles, r, m, stiffness):
        length = distance(start, end)
        unitStride = (end - start) / length
        self.gap = length / (num_particles - 1)
        
        self.particles = []
        for i in range(num_particles):
            pos = start + (i * self.gap * unitStride) 
            self.particles.append(Particle(pos, r, m))
        self.stiffness = stiffness

    def closest(self, x):
        minDist = None
        minI = None
        for i in range(len(self.particles)):
            particle = self.particles[i]
            dist = distance(x, particle.x)
            if minDist == None or dist < minDist:
                minDist = dist
                minI = i
        return minI
    
    def setFixed(self, i):
        if i < 0 or i >= len(self.particles): return
        self.particles[i].fixed = not self.particles[i].fixed

    def setPosition(self, i, pos):
        if i < 0 or i >= len(self.particles): return
        self.particles[i].x = pos
    
    def draw(self):
        for particle in self.particles:
            if particle.fixed: particle.draw()

        for i in range(len(self.particles)-1):
            startX, startY = self.particles[i].x
            endX, endY = self.particles[i+1].x
            drawLine(float(startX), float(startY), float(endX), float(endY))

    def add_gravity(self):
        for particle in self.particles:
            if not particle.fixed:
                particle.add_gravity()
    
    def update(self):
        for particle in self.particles:
            if not particle.fixed:
                particle.update()

    # https://owlree.blog/posts/simulating-a-rope.html
    def jakobsen(self):
        for _ in range(JAKOBSEN_ITERATIONS):
            for i in range(len(self.particles)-1):
                v = self.particles[i]
                w = self.particles[i+1]
                if v.fixed and w.fixed: continue  
                r = w.x - v.x
                r_mag = np.dot(r, r)**0.5
                r_hat = r / r_mag
                delta = r_mag - self.gap
                k = (1 - (1 - self.stiffness)**JAKOBSEN_ITERATIONS)
                if v.fixed:
                    w.x -= (k / w.m) * delta * r_hat
                elif w.fixed:
                    v.x += (k / v.m) * delta * r_hat
                else:
                    v.x += (k / v.m) * (delta / 2) * r_hat
                    w.x -= (k / w.m) * (delta / 2) * r_hat

def onAppStart(app):
    app.width, app.height = 600, 600
    app.stepsPerSecond = FPS
    reset(app)

def reset(app):
    app.paused = True
    app.dragIndex = None
    app.rope = None
    app.points = 0
    app.ropePoint1 = None
    app.ropePoint2 = None
    app.ropePoints = 20

def redrawAll(app):
    if app.rope != None:
        app.rope.draw()
    else:
        # draws rope preview
        if app.points >= 1:
            x1, y1 = app.ropePoint1
            drawCircle(float(x1), float(y1), 5, fill='blue')
        if app.points >= 2:
            x2, y2 = app.ropePoint2
            drawCircle(float(x2), float(y2), 5, fill='blue')
            drawLine(float(x1), float(y1), float(x2), float(y2))
            xStride = (x2 - x1) / (app.ropePoints - 1)
            yStride = (y2 - y1) / (app.ropePoints - 1)
            for i in range(1, app.ropePoints-1):
                drawCircle(float(x1 + i*xStride), float(y1 + i*yStride), 3, fill='blue')

    
    drawLabel(f"Number of Rope Points: {app.ropePoints}", 25, 25, size=20, align='left')

    pauseStr = "Paused" if app.paused else "Not Paused"
    drawLabel(pauseStr, 25, 575, size=20, align='left')


def onKeyPress(app, key):
    if key == 'p':
        app.paused = not app.paused
    elif key == 'r':
        reset(app)
    elif key == 'up':
        app.ropePoints += 1
    elif key == 'down':
        app.ropePoints -= 1
    elif key == 'enter' and app.rope == None and app.points == 2:
        app.rope = Rope(app.ropePoint1, app.ropePoint2, app.ropePoints, 5, 5, 1)

def onMousePress(app, mouseX, mouseY, button):
    if button == 0:
        if app.points == 0:
            app.ropePoint1 = vec(mouseX, mouseY)
            app.points += 1
        elif app.points == 1:
            app.ropePoint2 = vec(mouseX, mouseY)
            app.points += 1
        elif app.rope != None:
            closestIndex = app.rope.closest(vec(mouseX, mouseY))
            app.rope.setFixed(closestIndex)
    elif button == 2 and app.rope != None:
        app.dragIndex = app.rope.closest(vec(mouseX, mouseY))
        app.rope.setFixed(app.dragIndex)

def onMouseDrag(app, mouseX, mouseY, buttons):
    if app.rope == None: return
    if 2 in buttons and app.dragIndex != None:
        app.rope.setPosition(app.dragIndex, vec(mouseX, mouseY))

def onMouseRelease(app, mouseX, mouseY, button):
    if app.rope == None: return
    if button == 2 and app.dragIndex != None:
        app.rope.setFixed(app.dragIndex)
        app.dragIndex = None

def onStep(app):
    if app.paused: return
    
    if app.rope != None:
        app.rope.add_gravity()
        app.rope.update()
        app.rope.jakobsen()

def main():
    runApp()

if __name__ == "__main__":
    main()
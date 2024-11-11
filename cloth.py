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

class Cloth:

    def __init__(self, start, end, num_particles, r, m, stiffness):
        startX, startY = start
        endX, endY = end

        xLength = abs(endX - startX)
        yLength = abs(endY - startY)
        self.xGap = xLength / (num_particles - 1)
        self.yGap = yLength / (num_particles - 1)
        
        self.particles = [[None for _ in range(num_particles)] for _ in range(num_particles)]
        for i in range(num_particles):
            for j in range(num_particles):
                posX = startX + i * self.xGap
                posY = startY + j * self.yGap
                self.particles[i][j] = Particle(vec(posX, posY), r, m)

        self.stiffness = stiffness

    def closest(self, x):
        minDist = None
        minPos = None

        rows = len(self.particles)
        cols = len(self.particles[0])
        for i in range(rows):
            for j in range(cols):
                particle = self.particles[i][j]
                dist = distance(x, particle.x)
                if minDist == None or dist < minDist:
                    minDist = dist
                    minPos = i, j
        return minPos
    
    def setFixed(self, i, j):
        if i < 0 or i >= len(self.particles): return
        if j < 0 or j >= len(self.particles[0]): return
        self.particles[i][j].fixed = not self.particles[i][j].fixed

    def setPosition(self, i, j, pos):
        if i < 0 or i >= len(self.particles): return
        if j < 0 or j >= len(self.particles[0]): return
        self.particles[i][j].x = pos
    
    def draw(self):
        rows = len(self.particles)
        cols = len(self.particles[0])
        for i in range(rows):
            for j in range(cols):
                particle = self.particles[i][j]
                if particle.fixed: particle.draw()

        
        for i in range(rows - 1):
            for j in range(cols):
                startX, startY = self.particles[i][j].x
                endX, endY = self.particles[i+1][j].x
                drawLine(float(startX), float(startY), float(endX), float(endY))

        for i in range(rows):
            for j in range(cols - 1):
                startX, startY = self.particles[i][j].x
                endX, endY = self.particles[i][j+1].x
                drawLine(float(startX), float(startY), float(endX), float(endY))

    def add_gravity(self):
        rows = len(self.particles)
        cols = len(self.particles[0])
        for i in range(rows):
            for j in range(cols):
                particle = self.particles[i][j]
                particle.add_gravity()
    
    def update(self):
        rows = len(self.particles)
        cols = len(self.particles[0])
        for i in range(rows):
            for j in range(cols):
                particle = self.particles[i][j]
                if not particle.fixed:
                    particle.update()

    # https://owlree.blog/posts/simulating-a-rope.html
    def relax_constraint(self, v, w, gap):
        if v.fixed and w.fixed: return
        r = w.x - v.x
        r_mag = np.dot(r, r)**0.5
        r_hat = r / r_mag
        delta = r_mag - gap
        k = (1 - (1 - self.stiffness)**JAKOBSEN_ITERATIONS)
        if v.fixed:
            w.x -= (k / w.m) * delta * r_hat
        elif w.fixed:
            v.x += (k / v.m) * delta * r_hat
        else:
            v.x += (k / v.m) * (delta / 2) * r_hat
            w.x -= (k / w.m) * (delta / 2) * r_hat


    def jakobsen(self):
        rows = len(self.particles)
        cols = len(self.particles[0])
        for _ in range(JAKOBSEN_ITERATIONS):

            # vertical
            for i in range(rows - 1):
                for j in range(cols):
                    v = self.particles[i][j]
                    w = self.particles[i+1][j]
                    self.relax_constraint(v, w, self.yGap)

            # horizontal
            for i in range(rows):
                for j in range(cols - 1):
                    v = self.particles[i][j]
                    w = self.particles[i][j+1]
                    self.relax_constraint(v, w, self.xGap)

def onAppStart(app):
    app.width, app.height = 600, 600
    app.stepsPerSecond = FPS
    reset(app)

def reset(app):
    app.paused = True
    app.dragIndex = None
    app.cloth = None
    app.points = 0
    app.clothPoint1 = None
    app.clothPoint2 = None
    app.clothPoints = 20

def redrawAll(app):
    if app.cloth != None:
        app.cloth.draw()
    else:
        # draws cloth preview
        if app.points >= 1:
            x1, y1 = app.clothPoint1
            drawCircle(float(x1), float(y1), 5, fill='blue')
        if app.points >= 2:
            x2, y2 = app.clothPoint2
            drawCircle(float(x2), float(y2), 5, fill='blue')
            drawLine(float(x1), float(y1), float(x2), float(y1))
            drawLine(float(x1), float(y1), float(x1), float(y2))
            drawLine(float(x2), float(y1), float(x2), float(y2))
            drawLine(float(x1), float(y2), float(x2), float(y2))
    
    drawLabel(f"Number of Cloth Points: {app.clothPoints}", 25, 25, size=20, align='left')

    pauseStr = "Paused" if app.paused else "Not Paused"
    drawLabel(pauseStr, 25, 575, size=20, align='left')


def onKeyPress(app, key):
    if key == 'p':
        app.paused = not app.paused
    elif key == 'r':
        reset(app)
    elif key == 'up':
        app.clothPoints += 1
    elif key == 'down':
        app.clothPoints -= 1
    elif key == 'enter' and app.cloth == None and app.points == 2:
        app.cloth = Cloth(app.clothPoint1, app.clothPoint2, app.clothPoints, 5, 5, 1)

def onMousePress(app, mouseX, mouseY, button):
    if button == 0:
        if app.points == 0:
            app.clothPoint1 = vec(mouseX, mouseY)
            app.points += 1
        elif app.points == 1:
            app.clothPoint2 = vec(mouseX, mouseY)
            app.points += 1
        elif app.cloth != None:
            i, j = app.cloth.closest(vec(mouseX, mouseY))
            app.cloth.setFixed(i, j)
    elif button == 2 and app.cloth != None:
        app.dragIndex = app.cloth.closest(vec(mouseX, mouseY))
        app.cloth.setFixed(app.dragIndex[0], app.dragIndex[1])

def onMouseDrag(app, mouseX, mouseY, buttons):
    if app.cloth == None: return
    if 2 in buttons and app.dragIndex != None:
        app.cloth.setPosition(app.dragIndex[0], app.dragIndex[1], vec(mouseX, mouseY))

def onMouseRelease(app, mouseX, mouseY, button):
    if app.cloth == None: return
    if button == 2 and app.dragIndex != None:
        app.cloth.setFixed(app.dragIndex[0], app.dragIndex[1])
        app.dragIndex = None

def onStep(app):
    if app.paused: return
    
    if app.cloth != None:
        app.cloth.add_gravity()
        app.cloth.update()
        app.cloth.jakobsen()

def main():
    runApp()

if __name__ == "__main__":
    main()
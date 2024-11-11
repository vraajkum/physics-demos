from cmu_graphics import *
import numpy as np
import math

# constants
G = 10000
PLANET_DENSITY = 0.1
VELOCITY_NORM = 0.25
FPS = 100
DT = 1/FPS

# creates vectors
def vec(x, y):
    return np.array([x, y])

class Planet:

    def __init__(self, x, v, m, r):
        self.m = m
        self.r = r
        self.x = x
        self.f = vec(0, 0)

        # used for Euler integration
        self.v = v

        # used for Verlet integration
        self.xPrev = x - (v * DT)
    
    # updates position with Euler integration
    def euler_update(self):
        a = self.f / self.m
        self.v = self.v + (a * DT)

        self.xPrev = self.x
        self.x = self.x + (self.v * DT)

        self.f = vec(0, 0)
    
    # updates position with Verlet integration
    # https://en.wikipedia.org/wiki/Verlet_integration
    def verlet_update(self):
        a = self.f / self.m
        temp = self.x 
        self.x = (2 * self.x) - self.xPrev + (a * DT**2)
        self.xPrev = temp

        self.v = (self.x - self.xPrev) / DT
        self.f = vec(0, 0)
    
    # adds gravity between 2 planets
    @staticmethod
    def add_gravity(p1, p2):
        d = p2.x - p1.x
        d_squared = np.dot(d, d)
        d_hat = d / (d_squared ** 0.5)
        f_g = G * p1.m * p2.m / d_squared
        p1.f = p1.f + (f_g * d_hat)
        p2.f = p2.f - (f_g * d_hat)

    def draw(self):
        x, y = self.x
        drawCircle(float(x), float(y), self.r)

def onAppStart(app):
    app.width, app.height = 600, 600
    app.stepsPerSecond = FPS
    app.euler = True
    app.paused = False
    reset(app)

def reset(app):
    app.planets = []
    resetNewPlanet(app)

def resetNewPlanet(app):
    app.newPlanetX, app.newPlanetY = None, None
    app.newPlanetR = None
    app.newPlanetVelX, app.newPlanetVelY = None, None

def redrawAll(app):
    for planet in app.planets:
        planet.draw()

    if app.newPlanetX != None:
        drawCircle(app.newPlanetX, app.newPlanetY, app.newPlanetR, fill='blue')
        drawLine(app.newPlanetX, app.newPlanetY, app.newPlanetVelX, app.newPlanetVelY, arrowEnd=True)

    labelStr = "Euler" if app.euler else "Verlet"
    drawLabel(labelStr, 25, 25, size=20, align='left')

    pauseStr = "Paused" if app.paused else "Not Paused"
    drawLabel(pauseStr, 25, 575, size=20, align='left')
    

def onMousePress(app, mouseX, mouseY):
    if app.newPlanetX == None:
        app.newPlanetX, app.newPlanetY = mouseX, mouseY
        app.newPlanetVelX, app.newPlanetVelY = mouseX, mouseY
        app.newPlanetR = np.random.randint(15, 35)
    else:
        planetX = vec(app.newPlanetX, app.newPlanetY)
        planetV = VELOCITY_NORM * (vec(app.newPlanetVelX, app.newPlanetVelY) - planetX)
        planetA = math.pi * app.newPlanetR**2
        planetM = PLANET_DENSITY * planetA
        app.planets.append(Planet(planetX, planetV, planetM, app.newPlanetR))
        resetNewPlanet(app)

def onMouseMove(app, mouseX, mouseY):
    if app.newPlanetX != None:
        app.newPlanetVelX, app.newPlanetVelY = mouseX, mouseY

def onKeyPress(app, key):
    if key == 'm':
        app.euler = not app.euler
    elif key == 'p':
        app.paused = not app.paused
    elif key == 'r':
        reset(app)
    elif key == '1':
        demo_config(app)

def demo_config(app):
    planet1 = Planet(vec(230, 300), vec(0, 65), 250, 30)
    planet2 = Planet(vec(370, 300), vec(0, -65), 250, 30)
    app.planets = [planet1, planet2]

def onStep(app):
    if app.paused: return

    for i in range(len(app.planets)-1):
        for j in range(i+1, len(app.planets)):
            Planet.add_gravity(app.planets[i], app.planets[j])

    for planet in app.planets:
        if app.euler:
            planet.euler_update()
        else: 
            planet.verlet_update()

def main():
    runApp()

if __name__ == "__main__":
    main()
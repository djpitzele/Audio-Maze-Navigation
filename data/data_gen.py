import numpy
from audio_cave_sim import AudioCaveSim

if __name__ == "__main__":
    for i in range(65):
        sim = AudioCaveSim(Nx=81, Ny=81, res=0.01)
        sim.save()
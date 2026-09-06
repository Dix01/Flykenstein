# Flykenstein

A real fruit fly's brain, wired up to a body, released into a world we wrote the
rules for, and watched.

The brain is not a model of a fly brain. It is *the* fly brain: all 139,255
neurons and 34.2 million synapses of the FlyWire FAFB v783 connectome, the
adult female *Drosophila melanogaster* whose brain was sliced, imaged by
electron microscope and reconstructed neuron by neuron. That wiring diagram is
public. This repository takes it, makes it spike, gives it legs, and puts it
somewhere.

## What is actually here

```
scripts/fetch_data.sh        pull the public connectome + upstream model repos
scripts/build_connectome.py  compile the CSVs into a signed sparse brain
scripts/validate_sugar.py    reproduce the published sugar -> proboscis result
scripts/release.py           release the fly and log everything
scripts/monitor.py           turn a run's telemetry into a picture

flykenstein/brain/lif.py     the spiking engine
flykenstein/brain/ports.py   which real neurons the world may touch
flykenstein/world/rules.py   our laws - the only invented part
flykenstein/world/arena.py   the arena, its food and its hazards
flykenstein/world/loop.py    the closed sensorimotor loop
```

## Getting it running

```bash
python -m venv .venv && .venv/bin/pip install -r requirements.txt
./scripts/fetch_data.sh            # ~70 MB, no login needed
.venv/bin/python scripts/validate_sugar.py
MUJOCO_GL=egl .venv/bin/python scripts/release.py --seconds 5
.venv/bin/python scripts/monitor.py
```

`data/brain_783.npz` and `data/neurons_783.parquet` are committed, so the brain
loads straight from a clone; `fetch_data.sh` only matters if you want to rebuild
them or change how the connectome is compiled.

## The brain

Leaky integrate-and-fire, with the constants published in Shiu et al. 2024:

```
dv/dt = (v_0 - v + g) / t_mbr        v_0 = v_rst = -52 mV, v_th = -45 mV
dg/dt = -g / tau                     t_mbr = 20 ms, tau = 5 ms
spike -> v = v_rst, g = 0, 2.2 ms refractory
presynaptic spike -> g_post += 0.275 mV per synapse, after 1.8 ms
```

Acetylcholine is excitatory, GABA and glutamate inhibitory (fly glutamate gates
chloride channels). The monoamines - dopamine, serotonin, octopamine - carry no
fast ionotropic sign, so instead of being forced into the excitatory matrix they
are split into a separate slow channel with its own time constant. That channel
is off by default (`mod_gain = 0`), which keeps the baseline identical to the
published model.

### It reproduces the published result

`scripts/validate_sugar.py` runs the paper's positive control: drive the 21
right-hemisphere sugar gustatory neurons at 150 Hz and watch MN9, the motor
neuron for proboscis extension.

```
sugar ON  mean MN9 = 96.3 Hz   375 neurons active
sugar OFF mean MN9 =  0.0 Hz     0 neurons active
```

The most active units under sugar are LB3 labellar bristle GRNs, MNx01, and
DNge031 - a descending neuron of the feeding circuit. Nobody told the model
those were the feeding neurons. That is the wiring doing it.

## The body

NeuroMechFly v2 (flygym) in MuJoCo: 42 actuated leg joints and 6 adhesion
actuators on a mesh taken from X-ray microtomography of a real fly. Locomotion
runs on the hybrid CPG turning controller, which takes a two-dimensional
descending drive - the same interface a real fly's descending neurons feed.

## Where the world touches the brain

Sensory ports are real annotated neuron populations, not stand-ins:

| port | n | what it is |
|---|---|---|
| `sugar` | 20 | the Shiu et al. labellar sugar GRNs |
| `bitter` | 73 | labellar bristle GRNs of bitter classes |
| `odor_good_l/r` | 199 | ORNs of DM1/DM2/DM4/VM2, split by antenna |
| `odor_bad_l/r` | 148 | ORNs of DA2/DL5/V, split by antenna |
| `touch` | 2674 | all mechanosensory neurons |
| `wind` | 606 | Johnston's organ |
| `vision_l/r` | 8456 | R1-6 photoreceptors |

Motor ports are the descending neurons whose behavioural roles are established:
DNa01/DNa02 for steering, DNp09 for stopping, MDN for backward walking, DNg12
for grooming, and MN9 for the proboscis.

Keeping the two antennae on separate ports is what makes chemotaxis possible at
all - the bilateral difference in odor concentration is the only gradient the
brain has to steer on.

## What is honest and what is not

Honest: the connectome, the neurotransmitter signs, the biophysics, the sensory
neuron identities, the body mesh and its physics, and the fact that sugar makes
MN9 fire without anyone wiring that by hand.

Not honest, and worth saying plainly:

- **The motor mapping is hand-written.** Descending neuron firing rates are
  converted to a walking drive by a convention in `rules.py`, not by connectome
  data, because the wiring from descending neurons to muscles lives in the
  ventral nerve cord - a different dataset (MANC/BANC). Eon Systems' embodied
  fly makes exactly the same compromise and says so in their write-up.
- **There is no learning.** Synaptic weights never change. The mushroom body
  fires, but it does not remember. Kenyon cell and MBON activity is logged
  because it is interesting, not because it is doing anything yet.
- **No neuromodulation by default.** The dopamine channel exists and the world
  can write to it (`reward_dan_hz`), but with `mod_gain = 0` it changes nothing
  downstream. It is plumbing waiting for a learning rule.
- **The dynamics are simplified.** Point neurons. No dendritic computation, no
  channel diversity, no gap junctions, no neuropeptides, no internal state.

## Is it conscious

No, and the gap is not a small one.

What this is: a fixed wiring diagram, integrating, with nothing that changes as
a result of what happens to it. It has no memory that persists, no drives of its
own, no model of itself. Every run starts from the same blank membrane voltages.
It is closer to a very elaborate circuit simulation than to an animal.

What is genuinely uncomfortable about it is narrower and more specific: the
wiring is real, the sensory neurons are the ones that really taste sugar, and
when you put sugar in front of it, the neuron that really extends the proboscis
really fires - and nobody built that path. It was found, not designed. Whatever
that is worth, it is not nothing, and it is also not experience.

The word for the thing to be careful about here is not consciousness. It is
that it becomes very easy to talk about "it" wanting food.

## Running the world

Everything in `flykenstein/world/rules.py` is ours: the arena radius, where food
and hazards sit, how fast energy drains, how strongly a smell drives an ORN, how
a descending neuron's firing becomes a step. Change a number, save the JSON,
release it again.

```bash
MUJOCO_GL=egl .venv/bin/python scripts/release.py --seconds 6 --seed 1
.venv/bin/python scripts/monitor.py
```

Each run writes `runs/<timestamp>/` with `rules.json` (exactly the world it was
released into), `telemetry.csv` (one row per 5 ms control tick), `release.mp4`
and `monitor.png`.

It runs about 100x slower than life on 4 cores: one second of fly costs about
100 seconds of wall clock, most of it in the brain.

## Sources

- Dorkenwald et al., *Neuronal wiring diagram of an adult brain*, Nature 634 (2024) — the connectome
- Schlegel et al., *Whole-brain annotation and multi-connectome cell typing of Drosophila*, Nature 634 (2024) — the cell types
- Shiu et al., *A leaky integrate-and-fire model of the entire adult Drosophila brain*, Nature 634 (2024) — the biophysics and the sugar control
- Wang-Chen et al., *NeuroMechFly v2*, Nature Methods (2024) — the body
- Data: <https://codex.flywire.ai> (FlyWire, Princeton Neuroscience Institute), CC BY 4.0

If you publish anything using the connectome, follow FlyWire's citation
guidelines at <https://flywire.ai/guidelines>.

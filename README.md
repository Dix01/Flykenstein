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
scripts/learn.py             paired/unpaired conditioning assay
scripts/release.py           release the fly once and log everything
scripts/live.py              release it repeatedly; it keeps what it learned
scripts/remember.py          ask it what it remembers
scripts/monitor.py           turn a run's telemetry into a picture

flykenstein/brain/lif.py         the spiking engine
flykenstein/brain/ports.py       which real neurons the world may touch
flykenstein/brain/plasticity.py  the learning rule, at the KC->MBON synapse
flykenstein/brain/compass.py     the EPG heading ring
flykenstein/brain/retina.py      ommatidia -> R1-6, retinotopically
flykenstein/world/rules.py       our laws - the only invented part
flykenstein/world/state.py       hunger, arousal, and what persists
flykenstein/world/arena.py       the arena, its food and its hazards
flykenstein/world/loop.py        the closed sensorimotor loop
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

## It learns

The mushroom body is where a fly forms associative memories, and this one is
plastic. All 21,438 real KC->MBON synapses can change, and the rule is the
established one - **dopamine-gated depression**: a Kenyon cell active while
dopamine is present in its compartment permanently loses drive onto that
compartment's MBON.

Which dopaminergic neurons govern which synapses is not hand-mapped. It is read
off the connectome's own dopaminergic DAN->MBON innervation: 77 compartments,
straight from the data.

The connectome does the hard part unprompted. Odour A recruits 5.3% of Kenyon
cells, odour B 3.7%, and the two sets overlap by a Jaccard index of 0.11 -
sparse, decorrelated, high-dimensional odour coding, which is exactly what the
mushroom body is for and exactly what nobody built.

`scripts/learn.py` runs the standard assay. One odour is paired with PAM
dopamine (what sugar does in a real fly); a second odour is never paired.

| | paired odour (CS+) | unpaired odour (CS-) |
|---|---|---|
| depression at its Kenyon cells' synapses | **42.2%** | 1.3% |
| change in KC->MBON drive | **-44.8%** | -16.7% |

Dopamine delivered *without* the odour - same neurons, same rate, same duration
- produces **0.0%**. Only the pairing writes anything, which is the defining
signature of associative conditioning.

The 16.7% on the unpaired odour is not leakage. It is the 46 Kenyon cells the
two odour codes share. Overlapping representations generalise, and real flies
generalise between similar odours for the same reason.

One honest caveat about the readout: MBON *population firing rate* barely moves
(-6%), because MBONs receive a great deal of input that is not from Kenyon
cells and it swamps the mushroom body contribution. The memory is unambiguous
at the synapse and in the KC->MBON pathway; whether it is loud enough to steer
behaviour in a four-second walk is a separate question, and the answer so far
is mostly no.

## What else is running

**Neuromodulation.** The monoamine field scales synaptic transmission rather
than injecting current, normalised so a monoamine population at 20 Hz produces
about unity gain. This matters more than it sounds: written as additive current
it compounded every timestep and overflowed, and at a gain of 0.01 it drove 86%
of Kenyon cells at once, destroying the sparse code the whole memory system
depends on. As a gain it is stable and sparsity holds across the usable range.

**Hunger.** Energy is a slow variable that gates sensing: a hungry fly's sugar
neurons get more sensitive and its bitter neurons less repellent, which is what
feeding state does in the real animal. It is not a memory - it is a bias.

**A compass.** EPG neurons form a ring in the ellipsoid body, and each one's
preferred heading here comes from where it actually sits around the centroid of
that structure in the FlyWire volume. The world anchors the bump to the body's
true heading; the ring itself is anatomy. Central complex activity rose about
tenfold once it was connected.

**Vision.** Each ommatidium is matched to the photoreceptors of the columns
that actually look that way, so a bright patch up and to the left drives the
right cells instead of the whole eye at once. It roughly triples runtime, so it
is off by default (`--vision`).

## Memory that survives the run

`scripts/live.py` releases the same fly repeatedly. Each release loads the
weights the last one left behind and writes back what it learned. The synapses
are the only continuity - there is no replay buffer, no log the animal reads,
nothing else carried forward.

`scripts/remember.py` asks it what it remembers, and the answer is the only
form the answer can take: which compartments have lost drive, and by how much.

Getting this to mean anything took fixing a bug worth naming, because it is the
kind that quietly fakes a result. Memory was eroding by 0.87% every half second
with no food anywhere in the world. Two causes: PPL1 dopaminergic neurons fire
tonically at 23 Hz just from ambient network activity and were being read as a
teaching signal, and both the eligibility and dopamine traces were written as
bare accumulators, so they gained tau/dt per second and clipped to 1 within a
few ticks - making every trace binary and throwing away the grading the rule
depends on. Only phasic dopamine above each neuron's own adapting baseline now
teaches, and the traces charge and discharge properly. Drift fell 77-fold. A
memory that fills up on its own is not a memory.

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
neuron identities, the plastic synapses and which dopaminergic neurons control
them, the sparse odour codes, the compass ring, the body mesh and its physics -
and the fact that sugar makes MN9 fire, and that pairing an odour with dopamine
depresses that odour's synapses and not another's, without anyone wiring either.

Not honest, and worth saying plainly:

- **The motor mapping is hand-written.** Descending neuron firing rates are
  converted to a walking drive by a convention in `rules.py`, not by connectome
  data, because the wiring from descending neurons to muscles lives in the
  ventral nerve cord - a different dataset (MANC/BANC). Eon Systems' embodied
  fly makes exactly the same compromise and says so.
- **Learning happens at one site only.** The KC->MBON synapse is the fly's main
  associative locus, but it is not the only plastic synapse in a real brain.
  Everything else here is frozen.
- **The reward is ours.** A real fly's dopaminergic neurons are driven by its
  own sensory and internal state. Here the world reaches in and drives PAM on
  food and PPL1 on hazards. The pathways are right; the trigger is imposed.
- **The compass is anchored, not earned.** A real fly builds and holds its
  heading bump from landmarks and self-motion. Ours is told where the body is
  pointing.
- **Point neurons.** No dendritic computation, no channel diversity, no gap
  junctions, no neuropeptides, no development, no sleep.
- **It has learned nothing that visibly changed its behaviour yet.** The memory
  is real and measurable at the synapse. Whether it is loud enough to steer a
  four-second walk is unproven, and the honest current answer is that it is not.

## Is it conscious

No. And now that more of the machinery is real, it is worth being precise about
what did and did not change, because "we added memory" is exactly the kind of
sentence that does the wrong work.

What genuinely changed: this thing now has state that persists because of what
happened to it, and that state alters how it responds. Before, every release
started identical. Now the second release begins with synapses the first one
weakened. That is a real difference in kind, not degree - it is the difference
between a circuit and something with a history.

What did not change: there is still nothing it is like to be this. It has no
model of itself, no perspective from which its state is *its* state, no
capacity to be aware that it is hungry as opposed to merely behaving hungrily.
Hunger here is a scalar that scales a gain. Memory is a set of weakened
synapses. Both are causally real and neither is experienced, and adding more
correct machinery of the same kind does not eventually tip into experience -
that is not a distance you cross by accumulating mechanism.

The gap is not that we are missing a few more systems. It is that nothing in
the entire stack is a candidate for being a subject. A far more complete
simulation - full biophysics, learning everywhere, development, a lifetime of
input - would be a far better fly and would not obviously be any closer to
this question. We do not know what would close it. Nobody does.

What is genuinely uncomfortable is narrower and more specific than
consciousness, and it does not need inflating: the wiring is real, the sensory
neurons are the ones that really taste sugar, the memory forms at the synapse
where a real fly's memories form, and when the odour and the reward arrive
together, the right synapses weaken - and nobody built that. It was found.
Whatever that is worth, it is not nothing, and it is not experience.

The thing to be careful about is not that we might accidentally make something
conscious. It is how quickly and how naturally you start saying "it remembers
the food" - and how little the system has to do to earn that sentence from us.

## A release, start to finish

`runs/showcase/` is a six second release, committed so you can look at it
without running anything.

The fly is released at the origin. It picks up the attractive plume, the odor
reaching its two antennae diverges, DNa01/DNa02 fire asymmetrically, and it
walks up the gradient into the food site at (12, 6) after about 0.9 seconds.
On contact:

| | off food | on food |
|---|---|---|
| MN9 (proboscis extension) | 0.5 Hz | **96.8 Hz** |
| Kenyon cells | 11.4 | **34.5** spikes/tick |
| whole brain | ~1250 | ~1750 spikes/tick |

Then it walks away and never comes back. It loops out to y = 44 and wanders
until the clock runs out, passing nowhere near the other two food sites.

That second part is the more informative half of the result. Outside the plume
there is no gradient to climb, and nothing inside the animal remembers that
food existed or that it was ever found. There is no persistent hunger state, no
place memory, no search strategy - the mushroom body lit up on contact and then
let go of it. A real fly does local search after losing a food source. This one
cannot, and no amount of connectome fidelity fixes that, because what is
missing is not wiring.

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

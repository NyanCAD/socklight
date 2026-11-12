# The I/Q convention

Every socklight device and helper follows this convention for optical ports.

## Ports

- An optical port `p` is two real nodes, `p_i` and `p_q`: the in-phase and quadrature
  parts of the complex baseband envelope. Terminals are listed `p_i p_q` for each port,
  with ports in `sorted()` order.
- The representation is the same in DC (steady state, wavelength sweeps), AC
  (small-signal modulation; omega is the electrical frequency) and transient.

## Waves

- Real reference resistance z0 = 1 ohm.
- `a = (V + I)/2`, `b = (V - I)/2`, with complex `V = V_i + j V_q` and `I` flowing into the
  device port.
- Optical power carried by a wave is `|a|^2`, in watts by definition of this
  representation (no factor 1/2).
- Phase: sax's convention, `exp(+j phi)` for propagation.
- A model with scattering matrix S is stamped as the admittance
  `Y = (I - S)(I + S)^-1`, as `[[Re Y, -Im Y], [Im Y, Re Y]]`.

## Helper devices (Verilog-A, `socklight.va_path(name)`)

- `optical_source (o_i, o_q)`, parameters `p = 1.0` (W) and `phase = 0.0` (rad): a Thevenin
  source of `2 sqrt(p) e^{j phase}` behind 1 ohm on each rail, so the wave it emits has
  `|a|^2 = p` into a matched load.
- `optical_term (o_i, o_q)`: matched termination, 1 ohm from each rail to ground.
- `photodetector (o_i, o_q, p, n)`, parameters `responsivity = 1.0` (A/W) and
  `dark_current = 0.0` (A): 1 ohm on each optical rail and
  `I(n, p) <+ responsivity (V_i^2 + V_q^2) + dark_current`, so the photocurrent leaves the
  device at `p`.

## Rejected alternative

One node per port, using AC analysis at a fixed omega (stamp Re Y resistively and Im Y
reactively). It takes over omega, so electrical AC and transient analyses break, and
`|E|^2` detection is impossible.

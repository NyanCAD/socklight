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
- To keep `I + S` invertible when S has an eigenvalue of exactly -1, the conversion uses a
  reference impedance nudged off the real axis, `z0 = 1 + 1e-12j` (conj(z0) on the
  diagonal): it solves `(z0 S + conj(z0) I) Y = I - S`. The error this adds is of order 1e-12.
  A port with `T = S_kk = -1` (an ideal short) is therefore stamped as an admittance of about
  `1e12`, a very good conductor, not a true short: the node voltage is ~1e-12 of the drive
  rather than exactly 0.

## Helper devices (Verilog-A, `socklight.va_path(name)`)

- `optical_source (o_i, o_q)`, parameters `p = 1.0` (W) and `phase = 0.0` (rad): a Thevenin
  source of `2 sqrt(p) e^{j phase}` behind 1 ohm on each rail, so the wave it emits has
  `|a|^2 = p` into a matched load.
- `optical_term (o_i, o_q)`: matched termination, 1 ohm from each rail to ground.
- `photodetector (o_i, o_q, p, n)`, parameters `responsivity = 1.0` (A/W) and
  `dark_current = 0.0` (A): 1 ohm on each optical rail and
  `I(n, p) <+ responsivity (V_i^2 + V_q^2) + dark_current`, so the photocurrent leaves the
  device at `p`.

- `heater (o1_i, o1_q, o2_i, o2_q, l_e2, r_e2)`: thermal phase shifter (mirrors cspdk's
  `ThermalPhaseShifter`). Parameters: `ohms_per_um = 0.375`, `eta_pi_per_W = 12.5` (pi/W),
  `length = 320` (um), `loss_dBcm = 3.0`, `neff0 = 2.38`, `ng = 4.30`, `wl0 = 1.55` (um) and
  `wl = 1.55` (um).
  - Electrical: a resistor `R = ohms_per_um length` between `l_e2` and `r_e2`. The heater
    power `V^2/R` adds the phase `dphi = pi eta_pi_per_W V^2 / R`.
  - Optical: reciprocal and reflectionless, `S11 = S22 = 0` and `S21 = S12 = T`, with
    `T = 10^(-loss_dBcm length / 1e4 / 20) e^{j phi}`,
    `phi = 2 pi neff length / wl + dphi` and `neff = neff0 - (wl - wl0)(ng - neff0)/wl0`.
  - Formulation: written as scattering relations in the port currents,
    `V_k = I_k + T (V_j + I_j)` for `k != j` (from `V = a + b`, `I = a - b`, `b_k = T a_j`),
    split into I and Q. For this S, `det(I + S) = 1 - T^2`, so the admittance form
    `(I - S)(I + S)^-1` would be singular at `T = +-1`; this form is not.

## Rejected alternative

One node per port, using AC analysis at a fixed omega (stamp Re Y resistively and Im Y
reactively). It takes over omega, so electrical AC and transient analyses break, and
`|E|^2` detection is impossible.

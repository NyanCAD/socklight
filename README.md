# socklight

sax's optical models on numpy, exposed to VACASK as OSDI devices. See docs/design.md.

## Examples and cross-checks

- `examples/mzi_tia.py`: heater-tuned MZI into a photodetector and a TIA built from VACASK's bundled
  `opamp`; sweeping the heater voltage 0 to 4 V swings the TIA output between about 0 V and -4.9 V.
- `tools/circulax_heater.py` (run with `~/code/gdsfactory/cspdk/.venv/bin/python`): the same heater MZI in
  circulax with cspdk's `ThermalPhaseShifter`. It writes `tests/data/circulax_heater.json`;
  `tests/test_circulax.py` compares it with the VACASK run of the socklight heater. Max absolute
  difference of the two detector powers over an 11-point sweep (0 to 2 V): 4.3e-9 (circulax against
  a closed-form numpy MZI: 4.3e-9). Agreement is expected for the forward configuration only: circulax's
  heater is a VCVS and is non-reciprocal at `o2`, so reverse drive is not compared.

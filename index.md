# Spider modal analysis

Natural vibration modes of a pin-jointed spider: 17 parts joined by 16 pins and
standing on 8 pinned feet. The pins and feet are modelled with the RBE2, RBE3 and
spring features of [dolfinx_mpc](https://github.com/jorgensd/dolfinx_mpc), and the
lowest modes are found with a shift-and-invert eigensolver in SLEPc.

Pick a mode under **Warp**, tick **Animate** to see it oscillate, and use **Clip**
to look inside. The [implementation](spider_modes) shows how the model is built.

```{glue:} scene_view
:doc: spider_modes.py
```

```{glue:} frequency_table
:doc: spider_modes.py
```

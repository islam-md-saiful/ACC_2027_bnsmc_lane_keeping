# acc2027_bnsmc_lane_keeping

```
acc2027_bnsmc_lane_keeping/
├── models/
│   └── bntsmc_lane.onnx          # permanently trained model
│
├── scripts/
│   ├── train_lane_bnn.py         # run only once
│   └── lane_keeping_impl.py      # normal inference simulation
│
└── src/bntsmc/lane_keeping/
    ├── system.py
    ├── controller.py
    ├── experiment.py
    └── onnx_inference.py
    ```
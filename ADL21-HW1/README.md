# intent classification

### train script

```
python3.8 p1_train.py --data_dir ./data/intent/
```

### test script

```
python3.8 p1_test.py --data_dir ./data/intent/
```
or

```
./intent_cls.sh ./data/intent/test.json pred_intent.csv
```

# slot tagging

### train script

```
python3.8 p2_train.py --data_dir ./data/slot/
```

### test script

```
python3.8 p2_test.py --data_dir ./data/slot/
```
or

```
./slot_tag.sh data/slot/test.json pred_slot.csv
```
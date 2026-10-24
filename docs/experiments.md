# Experiments: what did not work, and what we learned

Model and input side; label experiments are in [labels.md](labels.md#what-failed). MEASURED unless tagged. "Fold
0" = one held-out fold of our 5-fold split, scored on our own labels; "the 58" = the expert-labelled training exams.

## Inputs

| Idea | Result; what we do now |
| --- | --- |
| Square 448 resize | Stretched 640 × 1280 images 2×; resize by spacing and pad |
| Longest series per plane | Non-fluid series in 5 of 12 slots with a fluid option; slots by plane × fluid flag |
| 150 mm centre crop in the store | Cut the field of view on 5,330 pilot slices; store the full view, crop in the model |
| 4 triplets per series | 15 triplets: +0.014 out of fold, +0.033 on the 58; every slice shown |
| 336 px at 140 mm | Seed noise, worse on Baker's; resolution pays only with a wider crop and stronger encoder |
| 384 px detail on enlarged boxes | −0.0009 on fold 0 at twice the cost: enlarging stored pixels adds nothing |
| Lateral close-up alone (84 mm, 384 px) | Lateral Meniscus −0.010, −0.0016 in a 50/50 mix; lateral-effnet keeps a whole-knee branch |
| Grey slab instead of 3-slice tiles | +0.0047 out of fold (bar 0.006), −0.002 on the 58; the forum's +0.020 (CLAIM) did not appear |
| Mirror augmentation on right-knee inputs | No gain; off |

## Models

| Idea | Result; what we do now |
| --- | --- |
| Mean or top-n pooling | Top-n lost twice (bundled with other changes); mean −0.014 to −0.02 (CLAIM, forum); attention pooling |
| DINOv2-S as a third family | Correlation with CoAtNet 0.867 (bar 0.746), below the CNNs at 336 px: errs like them |
| ResNet-50 next to CoAtNet | +0.0009 out of fold; the 58 and public unchanged |
| Knee-pretrained ViT-L, partly frozen | Fold 0 −0.0068, public 0.952; a small encoder trained end to end won |
| CoAtNet on detail crops, 224 px | Public 0.953, below the 0.956 bar |
| Whole knee + detail box in one model, 288 px | Fold 0 +0.0020 but public 0.959 against 0.964 |
| Place pooling + key-slice teaching + grade heads together | Fold 0 −0.0016 and −0.0020 (2 seeds); says nothing about each change |
| Fine-tuning a public 0.949 checkpoint | 0.959 alone, but too like our models: no blend gain |
| fp16 for window-effnet | NaN at step 1831, past the lr peak; bf16 |
| Under-trained CoAtNet | +5 epochs: +0.0065 on fold 0; rule out under-training first |

## Inference

| Idea | Result; what we do now |
| --- | --- |
| 13 models in one process | 0.586: out of memory on the hidden test, Kaggle scored a leftover file; one process per member, memory test on about 30 exams |
| A new member's first submission | 0.500: readers outran the GPU, rows piled up (IDEA); bounded hand-out, stress test on 200 exams |
| bf16 on T4 | About 50× slower (CLAIM); a rival's time guard filled half the test with 0.5; fp16 with fp32 rescue |
| Retry loops around `kaggle competitions submit` | One file sent 5 times, a day's slots used; submit once, list before retrying |

## Lessons

- **What the model sees beats which backbone:** every slice, detail crops, whole knee beside the close-up, crops cut
  once from the scanner's pixels. Backbone swaps mostly did not help.
- **Know each ruler:** fold 0 misranked families and cannot judge labels; the 58 (±0.02) misrank close models;
  public solo scores ranked families.
- **Fix the pass rule and blend weights before scoring:** weights searched on the 58 lost 0.002-0.007 to a fixed
  rule on the held-out half.
- **A weak test leaves an idea untested:** many "tried" verdicts were one fold, one seed, several changes.
- **Look at the inputs:** a pixel census found a double resize; an image sheet found the kneecap box too high.
- **The 3 visible test exams prove little:** they hid two out-of-memory failures.

## Never tried

- Mirror test-time augmentation (M): needs each series' left-right axis; others claimed +0.001.
- One model per plane (M): coronal for ACL and MCL, sagittal for the menisci.
- Mixup or CutMix (S): others claimed +0.002.
- More OAI MOAKS grades (L), also for Lateral OA and Synovitis: a rival claimed +0.005 public.
- A fast student of the blend (M): one fast mil-reader checkpoint scored 0.962 public (CLAIM, Matthew's run).

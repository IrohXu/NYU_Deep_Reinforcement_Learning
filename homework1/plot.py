"""
[CSCI-GA 3033-090] Special Topics: Deep Reinforcement Learning

Homework - 1, DAgger
Deadline: Sep 17, 2021 11:59 PM.

Plot for question 2
"""

import numpy as np
import matplotlib.pyplot as plt

# Use data from .plot_info/10250316_drl_hw1_xc2057.out

expert_queries = [1510,  3020,  4530,  6040,   7550,   9060,  10570,  12080,  13590,  15100,
  16610,  18120,  19630,  21140,  22650,  24160,  25670,  27180,  28690,  30200,
  31710,  33220,  34730,  36240,  37750,  39260,  40770,  42280,  43790,  45300,
  46810,  48320,  49830,  51340,  52850,  54360,  55870,  57380,  58890,  60400,
  61910,  63420,  64930,  66440,  67950,  69460,  70970,  72480,  73990,  75500,
  77010,  78520,  80030,  81540,  83050,  84560,  86070,  87580,  89090,  90600,
  92110,  93620,  95130,  96640,  98150,  99660, 101170, 102680, 104190, 105700,
 107210, 108720, 110230, 111740, 113250, 114760, 116270, 117780, 119290, 120800,
 122310, 123820, 125330, 126840, 128350, 129860, 131370, 132880, 134390, 135900,
 137410, 138920, 140430, 141940, 143450, 144960, 146470, 147980, 149490, 151000 ]


eval_rewards = [-1.14830695e+01, -5.59677487e-01,  1.72374746e+00,  4.66831844e-01,
 -5.53784846e+00, -7.68956604e+00, -3.19428710e-01, -8.91128962e+00,
 -8.29540889e+00, -7.90369022e+00, -2.54359430e+01, -1.48536148e+01,
 -1.55600016e+01,  1.97123162e+00, -3.10558203e+01, -1.70290962e+01,
 -3.49103667e+00, -7.18117957e+00,  6.80766239e-01, -2.79513589e+00,
 -4.21932773e-01, -1.85957260e+00,  1.91700791e+00, -2.09405590e+00,
 -6.11562627e+00, -1.67073654e+01, -1.83150657e+00,  2.04714433e+00,
  1.82003348e+01, -7.39121088e+00, -2.59264378e+01, -2.25815129e+01,
 -2.97619071e+01, -7.70963070e+00, -4.08297181e+00, -1.34899915e+01,
 -1.34790541e+01, -7.72413130e+00, -1.03523349e+01,  2.14717962e-02,
 -1.96812837e+00, -6.90970127e+00, -1.98357826e+01, -2.27948388e+01,
 -1.43837559e+01, -6.75699300e+00, -1.44314612e+01, -1.34930434e+01,
 -1.32907902e+01, -6.56191938e+00, -1.69460146e+01, -4.65623145e-01,
 -2.67229321e+00, -1.35885267e+01, -5.02918318e+00, -2.95276599e+01,
 -2.04445892e+01, -7.73126529e+00, -7.89503935e+00, -1.01026474e+01,
 -2.59360569e+01, -1.85270857e+01, -2.18800606e+01, -3.96573269e+00,
 -3.19501919e+01, -2.36780107e+01, -4.24473247e+01, -1.49359752e+01,
 -1.68299316e+01, -1.56730556e+01, -1.52107513e+01, -3.15863097e+01,
 -8.16485376e+00, -2.01282428e+01, -2.59832554e+01, -1.90568744e+01,
 -2.69029358e+01, -2.49550639e+01, -4.18934842e+01, -3.78983867e+01,
 -2.65160145e+01, -1.01672454e+01, -9.94336368e+00, -1.55363091e+01,
 -3.48382545e+01,  5.48361047e+00, -3.19959887e+01, -2.73468601e+01,
  6.20692540e+00, -1.61636059e+01,  5.70443630e+00, -5.23592479e+00,
 -1.87929654e+01,  6.53421108e+00, -1.21672287e+00,  1.13458812e+00,
 -1.63151553e+01, -3.18307743e+01, -2.70230329e+01, -1.91188397e+01 ]

# expert_queries = [  1510,   3020,   4530,   6040,   7550,   9060,  10570,  12080,  13590,  15100,
#   16610,  18120,  19630,  21140,  22650,  24160,  25670,  27180,  28690,  30200,
#   31710,  33220,  34730,  36240,  37750,  39260,  40770,  42280,  43790,  45300,
#   46810,  48320,  49830,  51340,  52850,  54360,  55870,  57380,  58890,  60400,
#   61910,  63420,  64930,  66440,  67950,  69460,  70970,  72480,  73990,  75500,
#   77010,  78520,  80030,  81540,  83050,  84560,  86070,  87580,  89090,  90600,
#   92110,  93620,  95130,  96640,  98150,  99660, 101170, 102680, 104190, 105700,
#  107210, 108720, 110230, 111740, 113250, 114760, 116270, 117780, 119290, 120800,
#  122310, 123820, 125330, 126840, 128350, 129860, 131370, 132880, 134390, 135900,
#  137410, 138920, 140430, 141940, 143450, 144960, 146470, 147980, 149490 ]

# eval_rewards = [ -6.63425758,   0.87849696, -21.11752745, -14.19552316,  -8.38356415,
#  -17.93077136,   0.98720171,   2.07022894,  -7.16789617, -11.34866399,
#  -18.03127174, -18.52766437, -10.7314515,    5.5307026,    0.98834372,
#   -9.9452183,   -5.45525908,   0.4777174,   -6.33674555, -14.81195557,
#   -2.4711223,    7.17875428, -17.41913131,  -2.98931662, -10.43462512,
#  -12.78488544, -12.08435589,   1.91292137, -14.14674911,  -8.77665183,
#   -6.98201602, -15.23311395,  -5.73556156,  -7.88777875,  -8.7552051,
#   -2.13819393, -13.84272432,  -9.25156112,   4.34554617, -11.01778915,
#   -4.62830411,  -5.39559348,   0.20504106, -21.29563835,  -7.60824697,
#    0.06465076,  -8.17179461,  -8.03043247,   1.8892416,   -5.35090788,
#    9.22540917,  -1.74390318,  -5.86609324, -12.42195085,  -1.44046189,
#   -2.59425429,  -5.87773334,  -6.76713577,   0.86091315,   3.75359992,
#    3.00705815,  -3.38837635,  10.97961569,  -7.87340639,  -4.87595714,
#   -9.88904187,  13.20729072,   2.30862238, -11.33121335, -21.85435406,
#   -7.89357929,  -3.88496252,  -9.5760065,    1.39036489,   1.11550816,
#   -2.35426477, -23.66537406,  -4.69232202, -15.74386691,  -1.82110437,
#  -10.47360345,  -6.01150305, -22.01810189,  -8.78994126, -11.08559216,
#  -33.60509653,  -5.82175039, -12.7368236,  -16.04647623,  -9.98844607,
#  -24.58609458, -13.75699914,  -3.29280715,  -3.41176785,   5.45935431,
#    8.26177276,  -7.22501013, -17.09904662,  -4.71757428]


plt.plot(expert_queries, eval_rewards)
plt.xlabel('Expert Queries')
plt.ylabel('Eval Rewards')
plt.title('Expert Queries vs Eval Rewards for DAgger')

plt.savefig('./q3.png')

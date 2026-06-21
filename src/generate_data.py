import pandas as pd
import numpy as np
import os


BASE = os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))
)


DATA_PATH = os.path.join(
    BASE,
    "data",
    "raw_data.csv"
)



np.random.seed(42)


rows = 5000


df = pd.DataFrame({

"machine_id":
range(1,rows+1),


"temperature":
np.random.normal(70,10,rows),


"pressure":
np.random.normal(50,8,rows),


"vibration":
np.random.normal(3,1,rows),


"humidity":
np.random.normal(40,10,rows),


"runtime_hours":
np.random.randint(100,10000,rows)

})



df["failure"]=(

(df.temperature>85) |

(df.pressure>70) |

(df.vibration>5) |

(df.runtime_hours>8000)

).astype(int)



df.to_csv(
DATA_PATH,
index=False
)


print("raw_data.csv created")
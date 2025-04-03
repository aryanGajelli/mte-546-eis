init_vmax = 5.281856619559205
init_vmin = -0.3794504905378653


dmm0 = 0
sal0 = 0

dmm1 = 3.8073
sal1 = 3.814

if dmm1 > dmm0:
    dmm1, dmm0 = dmm0, dmm1
    sal1, sal0 = sal0, sal1

adc0 = (sal0 - init_vmin)/(init_vmax-init_vmin)
adc1 = (sal1 - init_vmin)/(init_vmax-init_vmin)

vmax_m_vmin = dmm0/(adc0 - adc1)
vmin = dmm0 - vmax_m_vmin*adc0
vmax = vmax_m_vmin + vmin

print(f"{vmax=}, {vmin=}")

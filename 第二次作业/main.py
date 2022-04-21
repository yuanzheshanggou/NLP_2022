## EM
import numpy as np
import tensorboardX

def main():
    s1, s2, p, q, r = 0.3, 0.6, 0.4, 0.6, 0.3
    total_num = 1000
    head_num = int(total_num * (s1*p + s2*q + (1-s1-s2)*r))
    tail_num = total_num - head_num
    samples = np.zeros(total_num)
    samples[:head_num] = 1
    iterations = 20
    writer = tensorboardX.SummaryWriter()
    e_s1, e_s2, e_p, e_q, e_r = 0.4, 0.5, 0.5, 0.7, 0.4
    for i in range(iterations):
        writer.add_scalars('s1', {'s1':s1, 'e_s1':e_s1}, i, )
        writer.add_scalars('s2', {'s2':s2, 'e_s2':e_s2}, i)
        writer.add_scalars('p', {'p':p, 'e_p':e_p}, i)
        writer.add_scalars('q', {'q':q, 'e_q':e_q}, i)
        writer.add_scalars('r', {'r':r, 'e_r':e_r}, i)
        p_samples = np.power(e_p, samples)*np.power(1-e_p, 1-samples)*e_s1 + \
                    np.power(e_q, samples)*np.power(1-e_q, 1-samples)*e_s2 + \
                    np.power(e_r, samples)*np.power(1-e_r, 1-samples)*(1-e_s1-e_s2)
        pi1 = np.power(e_p, samples)*np.power(1-e_p, 1-samples)*e_s1 / p_samples
        pi2 = np.power(e_q, samples)*np.power(1-e_q, 1-samples)*e_s2 / p_samples
        pi3 = np.power(e_r, samples)*np.power(1-e_r, 1-samples)*(1-e_s1-e_s2) / p_samples
        e_s1, e_s2  = pi1.sum()/total_num, pi2.sum()/total_num
        e_p, e_q, e_r = np.sum(pi1*samples)/pi1.sum(), np.sum(pi2*samples)/pi2.sum(), np.sum(pi3*samples)/pi3.sum()
        

    print(e_s1, e_s2, e_p, e_q, e_r)
    writer.close()

if __name__ == "__main__":
    main()

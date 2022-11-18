import os
import argparse


def parse_config():
    parser = argparse.ArgumentParser(description='arg parser')
    parser.add_argument('--input_image_dir', type=str, default='../doc/result/asr_image.txt',
                        help='input_image')

    parser.add_argument('--output_image_dir', type=str, default='../doc/result/asr_lidar.txt',
                        help='input_lidar')

    parser.add_argument('--iutput_audio_dir', type=str, default='../doc/result/asr_overleaf.txt',
                        help='output_dir')

    args = parser.parse_args()
    return args


def OR(A, B):
    return A | B


def AND(A, B):
    return A & B


def write_txt_OR(f, input_image, input_lidar, n, k, r, t):
    f1 = open(input_image, 'r')
    f2 = open(input_lidar, 'r')
    line1 = f1.readlines()
    line2 = f2.readlines()

    for i in range(n):
        line1i = line1[i + r].split('  ')
        line2i = line2[i + r].split('  ')
        for j in range(k):
            val1 = line1i[j + t]
            val2 = line2i[j + t]
            result = OR(int(val1), int(val2))
            f.write(str(result) + '  ')
        f.write("\n")



def write_txt_AND(f, input_image, input_lidar, n, k, r, t):
    f1 = open(input_image, 'r')
    f2 = open(input_lidar, 'r')
    line1 = f1.readlines()
    line2 = f2.readlines()

    for i in range(n):
        line1i = line1[i + r].split('  ')
        line2i = line2[i + r].split('  ')
        for j in range(k):
            val1 = line1i[j + t]
            val2 = line2i[j + t]
            result =AND(int(val1), int(val2))
            f.write(str(result) + '  ')
        f.write("\n")

def write_output_func(input_image, input_lidar, output_combine, args, n, k):
    open(output_combine, 'w').close()

    with open(output_combine, 'a') as f:
        f.write(
            "00 01 02 03 04 05 06 07 08 09 10 11 12 13 14 15 16 17 18 19 20 21 22 23 24 25 26 27 28 29 OGA success_attack OGA_pointpillar")
        f.write('\n')
        write_txt_OR(f, input_image, input_lidar, n, k, 2, 1)

        f.write(
            "00 01 02 03 04 05 06 07 08 09 10 11 12 13 14 15 16 17 18 19 20 21 22 23 24 25 26 27 28 29 ODA success_attack ODA_pointpillar")
        f.write('\n')
        write_txt_AND(f, input_image, input_lidar, n, k, 10, 1)


def main():
    args = parse_config()
    input_image = os.path.join(args.input_image)
    input_lidar = os.path.join(args.input_lidar)
    output_combine = os.path.join(args.output_dir)

    n = 5
    k = 30

    write_output_func(input_image, input_lidar, output_combine, args, n, k)
    print("done")

if __name__ == '__main__':
    main()

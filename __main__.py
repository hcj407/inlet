from argparse import ArgumentParser
from inlet import performance, optimize
import sys



def main():

    ap = ArgumentParser(prog='Inlet', description='Pybaram Approximation Performance Analysis',
                        epilog='Example : inlet performance -g 2d -d 5,6,-13 --pbr 3,6')
    sp = ap.add_subparsers(dest='cmd', help='sub-command-help')

    #Inlet Find Ramp Angle
    ap_perf = sp.add_parser("find", help="Find intake ramp angle")
    

    # Inlet Performance
    ap_perf = sp.add_parser("performance", help="Evaluate intake performance")
    ap_perf.add_argument('-g', '--gtype', type=lambda s: s.lower(), help='Select inlet type : 2d or axi')
    
    tp = lambda x: [float(x) for x in x.split(',')]
    ap_perf.add_argument('-d', '--delta', required=False, type=tp, help="(optional) Input ramp's angle [deg]")
    ap_perf.set_defaults(func=performance.run)

    ap_perf.add_argument('-p', '--pbr', required=False, type=tp, help="(optional) Input back pressure ratio")
    ap_perf.set_defaults(func=performance.run)

    # Inlet Optimize
    ap_opt = sp.add_parser("optimize", help="Optimize design parameter")
    ap_opt.add_argument('-g', '--gtype', type=lambda s: s.lower(), help='Select inlet type : 2d or axi')
    
    tp = lambda x: list(map(float, x.replace(" ", "").split(",")))
    ap_opt.add_argument('-d', '--delta', required=False, type=tp, help="(optional) Input ramp's angle [deg]")
    ap_opt.set_defaults(func=optimize.run)

    args = ap.parse_args()
    args.func(args)

if __name__ == "__main__":
    main()
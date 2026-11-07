"""Data preparation, model fitting and event-coverage evaluation."""
from pathlib import Path
import argparse
import importlib
import sys
from threadpoolctl import threadpool_limits


def main():
    parser=argparse.ArgumentParser(description='Lake chlorophyll monitoring')
    sub=parser.add_subparsers(dest='command',required=True)
    predict=sub.add_parser('predict');predict.add_argument('--rows',type=int);predict.add_argument('--year',type=int,choices=range(2021,2026))
    score=sub.add_parser('score');score.add_argument('--model-root',type=Path,required=True);score.add_argument('--output',type=Path,required=True);score.add_argument('--rows',type=int)
    evaluate=sub.add_parser('evaluate');evaluate.add_argument('--output',type=Path);evaluate.add_argument('--predictions-root',type=Path);evaluate.add_argument('--bootstrap',type=int,default=0)
    for name in ['build-development','build-evaluation']:
        command=sub.add_parser(name);command.add_argument('--output',type=Path,required=True)
        if name=='build-evaluation':command.add_argument('--year',type=int,choices=range(2021,2026),required=True)
    for name in ['build-domain','build-observations','build-cross-region-observations','build-cross-region','evaluate-cross-region','analyze-us','adapt-china','train-relative-events']:
        sub.add_parser(name)
    sub.add_parser('train-cross-region')
    features=sub.add_parser('build-features');features.add_argument('--kind',choices=['static','water_quality','biomass','meteorology','seasonal','all'],default='all')
    download=sub.add_parser('download-lagos');download.add_argument('--output',type=Path,required=True)
    for name in ['train','validate-us']:
        command=sub.add_parser(name);command.add_argument('--output',type=Path,required=True);command.add_argument('--device',choices=['cpu','cuda'],default='cuda')
    args=parser.parse_args()
    with threadpool_limits(limits=2):
        if args.command=='predict':
            from .prediction import replay
            if args.rows is not None and args.rows<1:parser.error('--rows must be positive')
            replay(args.rows,[args.year] if args.year else range(2021,2026))
        elif args.command=='score':
            from .prediction import score
            score(args.model_root,args.output,rows=args.rows)
        elif args.command=='evaluate':
            from .evaluation import evaluate
            if args.predictions_root and not args.output:parser.error('--predictions-root requires --output')
            evaluate(output=args.output,reps=args.bootstrap,predictions_root=args.predictions_root)
        elif args.command=='download-lagos':
            from .acquisition import download_lagos
            download_lagos(args.output)
        elif args.command in ['build-observations','build-features']:
            from .build_features import observations,build,seasonal_history
            if args.command=='build-observations':observations()
            else:
                for kind in ['static','water_quality','biomass','meteorology','seasonal'] if args.kind=='all' else [args.kind]:
                    if kind=='seasonal':seasonal_history()
                    else:build(kind)
        elif args.command=='build-cross-region-observations':
            from .cross_region_observations import build_observations
            build_observations()
        elif args.command=='build-cross-region':
            from .cross_region_data import build_inputs
            build_inputs()
        elif args.command=='train-cross-region':
            from .cross_region_training import train
            train()
        elif args.command=='evaluate-cross-region':
            from .cross_region_evaluation import evaluate
            evaluate()
        elif args.command=='adapt-china':
            from .china_adaptation import run
            run()
        elif args.command=='train-relative-events':
            from .relative_events import run
            run()
        elif args.command=='analyze-us':
            from .development_evaluation import assess
            from .us_analysis import run
            assess();run()
        elif args.command.startswith('build-'):
            name={'build-domain':'build_domain','build-development':'build_development','build-evaluation':'build_evaluation'}[args.command]
            module=importlib.import_module('.'+name,__package__)
            sys.argv=[args.command]
            if hasattr(args,'output'):sys.argv+=['--output-root',str(args.output.resolve())]
            if hasattr(args,'year'):sys.argv+=['--year',str(args.year)]
            module.main()
        elif args.command=='train':
            from .training import refit
            refit(args.output,args.device)
        elif args.command=='validate-us':
            from .training import validate
            from .development_evaluation import assess
            validate(args.output,args.device);assess(args.output)


if __name__=='__main__':main()

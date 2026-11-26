import argparse
import json
from pathlib import Path
from .pipeline import setup, evaluate, run_date
from .training import train

def main():
    p=argparse.ArgumentParser();p.add_argument('--root',default=str(Path(__file__).resolve().parents[1]))
    sub=p.add_subparsers(dest='command',required=True)
    sub.add_parser('train');sub.add_parser('report')
    e=sub.add_parser('evaluate');e.add_argument('--role',choices=['core','transfer'],default='core');e.add_argument('--conditional',action='store_true')
    e.add_argument('--ridge-only',action='store_true',help='Backtest only the frozen Ridge model without calling the model API')
    d=sub.add_parser('run-date');d.add_argument('date');d.add_argument('--role',choices=['core','transfer'],default='core');d.add_argument('--explain',action='store_true');d.add_argument('--ticker');d.add_argument('--rating',choices=['Strong Buy','Buy','Hold','Sell','Strong Sell'])
    args=p.parse_args();root,config=setup(args.root)
    if args.command=='train': result=train(config,root)
    elif args.command=='evaluate':result=evaluate(root,args.role,args.conditional,ridge_only=args.ridge_only)
    elif args.command=='run-date':result=run_date(root,args.date,args.role,args.explain,args.ticker,args.rating)
    else:
        from .report import report
        result=report(root)
    print(json.dumps(result,ensure_ascii=False,indent=2,default=str))

if __name__=='__main__':main()

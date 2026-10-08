import joblib
import pandas as pd
from aged.absa import AspectSentimentAnalyzer

from aged.ingestion import segment_sentences

bundle = joblib.load('data/amazon_aged_model.joblib')
analyzer = AspectSentimentAnalyzer()

test_reviews = [
    ('5-star with 2 goods, 1 bad', 'The camera photo quality is stunning and the screen display is gorgeous, though battery drains a bit fast.'),
    ('3-star with 2 goods, 2 bads', 'Screen display is clear and build quality is sturdy, but software is laggy with bloatware and price is overpriced.'),
    ('1-star with 0 goods, 2 bads', 'Overheating terribly and battery drain is horrific. Useless device.'),
    ('2-star with 1 good, 2 bads', 'The build design feels premium, but camera photos are blurry and performance suffers from constant lag.')
]

for title, text in test_reviews:
    sentences = segment_sentences(text)
    res = analyzer.analyze_review(sentences, text)
    print(f"\n=== {title} ===")
    print(f"Goods: {res['num_good_aspects']} ({res['good_aspects']}) | Bads: {res['num_bad_aspects']} ({res['bad_aspects']})")
    
    feats = pd.DataFrame([{col: 0.0 for col in bundle['feature_names']}])
    feats['num_good_aspects'] = float(res['num_good_aspects'])
    feats['num_bad_aspects'] = float(res['num_bad_aspects'])
    feats['goods_count'] = float(res['goods_count'])
    feats['bads_count'] = float(res['bads_count'])
    feats['net_aspect_balance'] = float(res['net_aspect_balance'])
    feats['aspect_valence_ratio'] = float(res['aspect_valence_ratio'])
    feats['goods_times_bads'] = float(res['num_good_aspects'] * res['num_bad_aspects'])
    feats['bads_squared'] = float(res['num_bad_aspects'] ** 2)
    feats['has_both_goods_and_bads'] = 1.0 if (res['num_good_aspects'] > 0 and res['num_bad_aspects'] > 0) else 0.0
    feats['only_bads'] = 1.0 if (res['num_good_aspects'] == 0 and res['num_bad_aspects'] > 0) else 0.0
    feats['only_goods'] = 1.0 if (res['num_good_aspects'] > 0 and res['num_bad_aspects'] == 0) else 0.0
    
    for g in res['good_aspects']:
        if f'good_{g}' in feats.columns:
            feats[f'good_{g}'] = 1.0
    for b in res['bad_aspects']:
        if f'bad_{b}' in feats.columns:
            feats[f'bad_{b}'] = 1.0
            
    feats['overall_text_sentiment'] = float(res['overall_text_sentiment'])
    feats['mean_aspect_sentiment'] = float(res['mean_aspect_sentiment'])
    feats['prior_rating_mean'] = 3.8
    
    pred_stars = bundle['regressor'].predict(feats)[0]
    probs = bundle['classifier'].predict_proba(feats)[0]
    classes = bundle['classifier'].classes_
    prob_dict = {f"{c}-star": round(p, 3) for c, p in zip(classes, probs)}
    print(f"Predicted Rating: {pred_stars:.2f} stars")
    print(f"Star Probabilities: {prob_dict}")

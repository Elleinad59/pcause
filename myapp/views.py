from django.shortcuts import render, redirect
from django.views.decorators.csrf import csrf_exempt
import joblib
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from lime.lime_tabular import LimeTabularExplainer
from io import BytesIO
import base64

# Feature mapping from form names to final dataset headers
yes_no_fields = ['hair-growth', 'weight-gain', 'skin-darkening', 'fast-food', 'pimples', 'hair-loss']

feature_names = [
    'hair_growth_y_n', 'weight_gain_y_n', 'skin_darkening_y_n', 'fast_food_y_n',
    'cycle_length_days', 'cycle_r_i', 'age_yrs', 'bmi',
    'pulse_rate_bpm', 'pimples_y_n', 'rr_breaths_min', 'hair_loss_y_n'
]

def landing_page(request):
    return render(request, 'myapp/landing.html')

@csrf_exempt
def take_test(request):
    if request.method == 'POST':
        try:
            def yn(value): return 1 if value == 'Yes' else 0
            def cycle_map(value): return 2 if value == 'Regular' else 4

            input_data = {
                'hair_growth_y_n': yn(request.POST.get('hair-growth')),
                'weight_gain_y_n': yn(request.POST.get('weight-gain')),
                'skin_darkening_y_n': yn(request.POST.get('skin-darkening')),
                'fast_food_y_n': yn(request.POST.get('fast-food')),
                'cycle_length_days': float(request.POST.get('cycle-length')),
                'cycle_r_i': cycle_map(request.POST.get('cycle')),
                'age_yrs': int(request.POST.get('age')),
                'bmi': float(request.POST.get('bmi')),
                'pulse_rate_bpm': int(request.POST.get('pulse')),
                'pimples_y_n': yn(request.POST.get('pimples')),
                'rr_breaths_min': int(request.POST.get('rr')),
                'hair_loss_y_n': yn(request.POST.get('hair-loss')),
            }

            input_df = pd.DataFrame([input_data])

            model_path = 'myapp/models/knn_config4_model.pkl'
            model = joblib.load(model_path)

            prediction = model.predict(input_df)[0]
            probability = model.predict_proba(input_df)[0][1]

            request.session['result'] = "PCOS" if prediction == 1 else "No PCOS"
            request.session['probability'] = round(probability * 100, 2)

            for key, value in input_data.items():
                request.session[key] = value

            return redirect('result_view')

        except Exception as e:
            return render(request, 'myapp/take_test.html', {
                'error': str(e),
                'yes_no_fields': yes_no_fields
            })

    return render(request, 'myapp/take_test.html', {
        'yes_no_fields': yes_no_fields
    })

def result_view(request):
    result = request.session.get('result')
    probability = request.session.get('probability')

    if not result:
        return redirect('take_test')

    feature_names = [
        'hair_growth_y_n', 'weight_gain_y_n', 'skin_darkening_y_n', 'fast_food_y_n',
        'cycle_length_days', 'cycle_r_i', 'age_yrs', 'bmi',
        'pulse_rate_bpm', 'pimples_y_n', 'rr_breaths_min', 'hair_loss_y_n'
    ]

    try:
        input_data = [
            int(request.session.get('hair_growth_y_n')),
            int(request.session.get('weight_gain_y_n')),
            int(request.session.get('skin_darkening_y_n')),
            int(request.session.get('fast_food_y_n')),
            float(request.session.get('cycle_length_days')),
            int(request.session.get('cycle_r_i')),
            int(request.session.get('age_yrs')),
            float(request.session.get('bmi')),
            int(request.session.get('pulse_rate_bpm')),
            int(request.session.get('pimples_y_n')),
            int(request.session.get('rr_breaths_min')),
            int(request.session.get('hair_loss_y_n')),
        ]
    except Exception as e:
        return render(request, 'myapp/results.html', {
            'result': result,
            'probability': probability,
            'lime_plot': None,
            'summary_text': f"Data conversion failed: {str(e)}"
        })

    input_df = pd.DataFrame([input_data], columns=feature_names)

    lime_plot = None
    summary_text = None

    try:
        model_path = 'myapp/models/knn_config4_model.pkl'
        model = joblib.load(model_path)
        preprocessor = model.named_steps['pre']
        classifier = model.named_steps['clf']

        background_df = pd.read_csv('myapp/models/final_lime_background.csv')

        explainer = LimeTabularExplainer(
            training_data=background_df.values,
            feature_names=feature_names,
            class_names=['No PCOS', 'PCOS'],
            mode='classification'
        )

        exp = explainer.explain_instance(
            data_row=input_df.iloc[0].values,
            predict_fn=lambda x: classifier.predict_proba(preprocessor.transform(pd.DataFrame(x, columns=feature_names)))
        )

        class_label = 1
        local_exp = exp.local_exp[class_label]

        exp_map = {i: w for i, w in local_exp}
        for i in range(len(feature_names)):
            if i not in exp_map:
                exp_map[i] = 0.0

        sorted_exp = sorted(exp_map.items(), key=lambda x: abs(x[1]), reverse=True)
        indices = [i for i, _ in sorted_exp]
        weights = [w for _, w in sorted_exp]
        features = [explainer.feature_names[i] for i in indices]
        colors = ['green' if w > 0 else 'red' for w in weights]

        fig, ax = plt.subplots(figsize=(10, 6))
        y_pos = np.arange(len(features))
        ax.barh(y_pos, weights, align='center', color=colors)
        ax.set_yticks(y_pos)
        ax.set_yticklabels(features)
        ax.invert_yaxis()
        ax.axvline(0, color='black', linewidth=0.8)
        ax.set_title("Local explanation for class PCOS")
        plt.tight_layout()

        buf = BytesIO()
        plt.savefig(buf, format='png', bbox_inches='tight')
        plt.close(fig)
        buf.seek(0)
        lime_plot = base64.b64encode(buf.read()).decode('utf-8')

        # Friendly labels for display
        friendly_labels = {
            'hair_growth_y_n': 'Excess hair growth',
            'weight_gain_y_n': 'Weight gain',
            'skin_darkening_y_n': 'Skin darkening',
            'fast_food_y_n': 'Frequent fast food intake',
            'cycle_length_days': 'Cycle length (days)',
            'cycle_r_i': 'Menstrual cycle pattern',
            'age_yrs': 'Age',
            'bmi': 'BMI (Body Mass Index)',
            'pulse_rate_bpm': 'Pulse rate (bpm)',
            'pimples_y_n': 'Pimples or acne',
            'rr_breaths_min': 'Respiratory rate (breaths/min)',
            'hair_loss_y_n': 'Hair thinning or loss'
        }

        # Summary text logic
        top_factors = []
        for i, w in sorted_exp[:3]:
            fname = explainer.feature_names[i]
            label = friendly_labels.get(fname, fname.replace('_', ' ').capitalize())
            direction = "Increased PCOS risk" if w > 0 else "Reduced PCOS risk"
            top_factors.append(f"{label} — <span style='font-style: italic;'>{direction}</span>")


        summary_text = """
<div style="margin-top: 1rem; font-weight: 400;">
    <div style="font-size: 1rem; margin-bottom: 0.5rem;">
         <span style="font-weight: 500;">Key factors influencing this prediction:</span>
    </div>
    <ul style="margin-top: 0.25rem; padding-left: 1.25rem; line-height: 1.6;">
        {}
    </ul>
    <p style="margin-top: 0.75rem; font-size: 0.95rem; color: #444;">
        Features marked as <em>reduced PCOS risk</em> suggest protective patterns, while those marked as <em>increased PCOS risk</em> indicate stronger associations with PCOS symptoms.
    </p>
</div>
""".format("".join(f"<li>{f}</li>" for f in top_factors))


    except Exception as e:
        lime_plot = None
        summary_text = None
        print("❌ LIME generation failed:", str(e))

    return render(request, 'myapp/results.html', {
        'result': result,
        'probability': probability,
        'lime_plot': lime_plot,
        'summary_text': summary_text
    })



def pcos_view(request):
    return render(request, 'myapp/pcos.html')

def about_view(request):
    return render(request, 'myapp/about.html')

# step_09_select_metric.py
scoring_metric = 'f1_macro'

print("="*60)
print("КРОК 9: ВИБІР МЕТРИКИ ДЛЯ ОПТИМІЗАЦІЇ")
print("="*60)
print(f"Обрана метрика: {scoring_metric}")
print("Пояснення:")
print("• F1-macro — середнє F1 по всіх класах")
print("• Чутлива до дисбалансу")
print("• Оптимальна для мультикласової класифікації")
print("="*60)
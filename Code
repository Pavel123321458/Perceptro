import numpy as np
import matplotlib.pyplot as plt


def make_custom_data(n_samples=500, data_type='linear', noise=0.0, random_state=42):
    np.random.seed(random_state)
    half = n_samples // 2

    X = np.zeros((n_samples, 2))
    y = np.zeros(n_samples, dtype=int)

    if data_type == 'linear':
        X[:half] = np.random.multivariate_normal([2, 2], [[1, 0.5], [0.5, 1]], half)
        y[:half] = 0
        X[half:] = np.random.multivariate_normal([-2, -2], [[1, -0.5], [-0.5, 1]], n_samples - half)
        y[half:] = 1

    elif data_type == 'xor':
        quarter = n_samples // 4
        X[0:quarter] = np.random.randn(quarter, 2) * 0.3 + [0, 1]
        y[0:quarter] = 0
        X[quarter:2 * quarter] = np.random.randn(quarter, 2) * 0.3 + [1, 0]
        y[quarter:2 * quarter] = 0
        X[2 * quarter:3 * quarter] = np.random.randn(quarter, 2) * 0.3 + [0, 0]
        y[2 * quarter:3 * quarter] = 1
        X[3 * quarter:4 * quarter] = np.random.randn(quarter, 2) * 0.3 + [1, 1]
        y[3 * quarter:4 * quarter] = 1

    elif data_type == 'circle':
        r = np.random.rand(half) * 0.8
        angle = np.random.rand(half) * 2 * np.pi
        X[:half, 0] = r * np.cos(angle)
        X[:half, 1] = r * np.sin(angle)
        y[:half] = 0
        r = 1.2 + np.random.rand(n_samples - half) * 1.0
        angle = np.random.rand(n_samples - half) * 2 * np.pi
        X[half:, 0] = r * np.cos(angle)
        X[half:, 1] = r * np.sin(angle)
        y[half:] = 1

    if noise > 0:
        flip = np.random.choice(n_samples, int(n_samples * noise), replace=False)
        y[flip] = 1 - y[flip]

    idx = np.random.permutation(n_samples)

    return X[idx], y[idx].reshape(-1, 1)

X, y = make_custom_data(n_samples=500, data_type='linear', noise=0.05, random_state=42)




from sklearn.datasets import make_classification
"""X, y = make_classification(
    n_samples=500,
    n_features=2,
    n_redundant=0,
    n_informative=2,
    random_state=42,
    n_clusters_per_class=1
)
y = y.reshape(-1, 1)"""


class StandardScaler:
    def fit(self, X):
        self.mean = X.mean(axis=0)
        self.std = X.std(axis=0) + 1e-8

    def transform(self, X):
        return (X - self.mean) / self.std

    def fit_transform(self, X):
        self.fit(X)
        return self.transform(X)


def train_test_split_stratified(X, y, test_size=0.3, random_state=42):
    np.random.seed(random_state)
    classes = np.unique(y)
    X_train, X_test, y_train, y_test = [], [], [], []
    for cls in classes:
        idx = np.where(y == cls)[0]
        np.random.shuffle(idx)
        split = int(len(idx) * (1 - test_size))
        X_train.append(X[idx[:split]])
        y_train.append(y[idx[:split]])
        X_test.append(X[idx[split:]])
        y_test.append(y[idx[split:]])
    return (np.vstack(X_train), np.vstack(X_test),
            np.concatenate([yt.ravel() for yt in y_train]),
            np.concatenate([yt.ravel() for yt in y_test]))


X_train, X_test, y_train, y_test = train_test_split_stratified(X, y, test_size=0.3)
X_train, X_val, y_train, y_val = train_test_split_stratified(X_train, y_train, test_size=0.2)

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_val = scaler.transform(X_val)
X_test = scaler.transform(X_test)


class Perceptron:
    def __init__(self, n_features):
        self.w = np.random.randn(n_features, 1) * 0.01
        self.b = 0.0
        self.train_losses = []
        self.val_losses = []

    def sigmoid(self, z):
        return 1.0 / (1.0 + np.exp(-z))

    def forward(self, X):
        return self.sigmoid(X @ self.w + self.b)

    def binary_cross_entropy(self, y_true, y_pred):
        eps = 1e-15
        y_pred = np.clip(y_pred, eps, 1 - eps)
        return -np.mean(y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred))

    def hinge_loss(self, y_true, z):
        """Hinge loss: y_true ∈ {-1, +1}, z — сырые значения (до сигмоиды)"""
        return np.mean(np.maximum(0, 1 - y_true * z))

    def fit(self, X_train, y_train, X_val, y_val, epochs=100, lr=0.1, batch_size=32):
        self.train_losses = []
        self.val_losses = []
        y_train = y_train.reshape(-1, 1)
        y_val = y_val.reshape(-1, 1)
        n = X_train.shape[0]

        for epoch in range(epochs):
            idx = np.random.permutation(n)
            X_shuffled = X_train[idx]
            y_shuffled = y_train[idx]

            for start in range(0, n, batch_size):
                end = min(start + batch_size, n)
                X_batch = X_shuffled[start:end]
                y_batch = y_shuffled[start:end]

                y_pred = self.forward(X_batch)
                error = y_pred - y_batch

                dw = X_batch.T @ error / len(X_batch)
                db = np.mean(error)

                self.w -= lr * dw
                self.b -= lr * db

            self.train_losses.append(self.binary_cross_entropy(y_train, self.forward(X_train)))
            self.val_losses.append(self.binary_cross_entropy(y_val, self.forward(X_val)))

    def fit_hinge(self, X_train, y_train, X_val, y_val, epochs=100, lr=0.1, batch_size=32):
        self.train_losses = []
        self.val_losses = []
        y_train = y_train.reshape(-1, 1)
        y_val = y_val.reshape(-1, 1)
        n = X_train.shape[0]

        for epoch in range(epochs):
            idx = np.random.permutation(n)
            X_shuffled = X_train[idx]
            y_shuffled = y_train[idx]

            for start in range(0, n, batch_size):
                end = min(start + batch_size, n)
                X_batch = X_shuffled[start:end]
                y_batch = y_shuffled[start:end]

                z = X_batch @ self.w + self.b
                margin = 1 - y_batch * z
                mask = (margin > 0).astype(float)
                dw = -X_batch.T @ (y_batch * mask) / len(X_batch)
                db = -np.mean(y_batch * mask)

                self.w -= lr * dw
                self.b -= lr * db

            z_train = X_train @ self.w + self.b
            z_val = X_val @ self.w + self.b
            self.train_losses.append(self.hinge_loss(y_train, z_train))
            self.val_losses.append(self.hinge_loss(y_val, z_val))

    def fit_l2(self, X_train, y_train, X_val, y_val, epochs=100, lr=0.1, batch_size=32, lambd=0.01):
        self.train_losses = []
        self.val_losses = []
        y_train = y_train.reshape(-1, 1)
        y_val = y_val.reshape(-1, 1)
        n = X_train.shape[0]

        for epoch in range(epochs):
            idx = np.random.permutation(n)
            X_shuffled = X_train[idx]
            y_shuffled = y_train[idx]

            for start in range(0, n, batch_size):
                end = min(start + batch_size, n)
                X_batch = X_shuffled[start:end]
                y_batch = y_shuffled[start:end]

                y_pred = self.forward(X_batch)
                error = y_pred - y_batch
                dw = X_batch.T @ error / len(X_batch) + lambd * self.w
                db = np.mean(error)

                self.w -= lr * dw
                self.b -= lr * db

            bce = self.binary_cross_entropy(y_train, self.forward(X_train))
            l2 = lambd * np.sum(self.w ** 2)
            self.train_losses.append(bce + l2)
            bce_val = self.binary_cross_entropy(y_val, self.forward(X_val))
            self.val_losses.append(bce_val + lambd * np.sum(self.w ** 2))

    def fit_momentum(self, X_train, y_train, X_val, y_val, epochs=100, lr=0.1, batch_size=32, beta=0.9):
        """SGD с моментом"""
        self.train_losses = []
        self.val_losses = []
        y_train = y_train.reshape(-1, 1)
        y_val = y_val.reshape(-1, 1)
        n = X_train.shape[0]

        v_w = np.zeros_like(self.w)
        v_b = 0.0

        for epoch in range(epochs):
            idx = np.random.permutation(n)
            X_shuffled = X_train[idx]
            y_shuffled = y_train[idx]

            for start in range(0, n, batch_size):
                end = min(start + batch_size, n)
                X_batch = X_shuffled[start:end]
                y_batch = y_shuffled[start:end]

                y_pred = self.forward(X_batch)
                error = y_pred - y_batch

                dw = X_batch.T @ error / len(X_batch)
                db = np.mean(error)

                v_w = beta * v_w + (1 - beta) * dw
                v_b = beta * v_b + (1 - beta) * db

                self.w -= lr * v_w
                self.b -= lr * v_b

            self.train_losses.append(self.binary_cross_entropy(y_train, self.forward(X_train)))
            self.val_losses.append(self.binary_cross_entropy(y_val, self.forward(X_val)))

    def predict(self, X):
        return (self.forward(X) >= 0.5).astype(int).ravel()

    def accuracy(self, X, y):
        return np.mean(self.predict(X) == y)


# ========== 1. Обучение ==========
model = Perceptron(n_features=2)
model.fit(X_train, y_train, X_val, y_val, epochs=100, lr=0.1, batch_size=32)

print(f"Train accuracy: {model.accuracy(X_train, y_train):.4f}")
print(f"Test accuracy:  {model.accuracy(X_test, y_test):.4f}")

# ========== 2. Графики ==========
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

axes[0].plot(model.train_losses, label='Train')
axes[0].plot(model.val_losses, label='Val')
axes[0].set_xlabel('Epoch')
axes[0].set_ylabel('Loss')
axes[0].set_title('Binary Cross-Entropy Loss')
axes[0].legend()
axes[0].grid()

axes[1].scatter(X_train[:, 0], X_train[:, 1], c=y_train, cmap='bwr', alpha=0.6, edgecolors='k')
axes[1].scatter(X_test[:, 0], X_test[:, 1], c=y_test, cmap='bwr', alpha=0.3, marker='s')
x1_range = np.linspace(X_train[:, 0].min(), X_train[:, 0].max(), 100)
x2_range = -(model.w[0, 0] * x1_range + model.b) / model.w[1, 0]
axes[1].plot(x1_range, x2_range, 'g-', lw=2, label='Decision boundary')
axes[1].set_xlabel('x1')
axes[1].set_ylabel('x2')
axes[1].set_title('Decision boundary')
axes[1].legend()
axes[1].grid()
plt.tight_layout()
plt.show()

# ========== 3. Эксперименты ==========
print("=== Влияние learning rate ===")
for lr in [0.001, 0.01, 0.5, 1.0]:
    m = Perceptron(2)
    m.fit(X_train, y_train, X_val, y_val, epochs=100, lr=lr, batch_size=32)
    print(f"lr={lr:.3f} -> test acc={m.accuracy(X_test, y_test):.4f}")

print("=== Влияние batch size ===")
for bs in [1, 16, 64, 256]:
    m = Perceptron(2)
    m.fit(X_train, y_train, X_val, y_val, epochs=100, lr=0.1, batch_size=bs)
    print(f"batch={bs:3d} -> test acc={m.accuracy(X_test, y_test):.4f}")

print("\n=== Влияние инициализации весов ===")
print("1. Нулевая инициализация:")
m = Perceptron(2)
m.w = np.zeros((2, 1))
m.b = 0.0
m.fit(X_train, y_train, X_val, y_val, epochs=100, lr=0.1, batch_size=32)
print(f"   Test acc = {m.accuracy(X_test, y_test):.4f}")

print("2. Маленькие случайные (N(0, 0.01)):")
m = Perceptron(2)         # и так внутри randn * 0.01
m.fit(X_train, y_train, X_val, y_val, epochs=100, lr=0.1, batch_size=32)
print(f"   Test acc = {m.accuracy(X_test, y_test):.4f}")

print("3. Большие случайные (N(0, 10)):")
m = Perceptron(2)
m.w = np.random.randn(2, 1) * 10   # меняем на большие
m.fit(X_train, y_train, X_val, y_val, epochs=100, lr=0.1, batch_size=32)
print(f"   Test acc = {m.accuracy(X_test, y_test):.4f}")

# ========== Доп. задание 2: Hinge loss и L2-регуляризация ==========
print("\n=== Доп. задание 2: Hinge loss ===")

y_train_hinge = y_train * 2 - 1
y_val_hinge = y_val * 2 - 1

model_hinge = Perceptron(2)
model_hinge.fit_hinge(X_train, y_train_hinge, X_val, y_val_hinge, epochs=100, lr=0.1, batch_size=32)
acc_hinge = model_hinge.accuracy(X_test, y_test)
print(f"Hinge loss -> test acc = {acc_hinge:.4f}")

model_bce = Perceptron(2)
model_bce.fit(X_train, y_train, X_val, y_val, epochs=100, lr=0.1, batch_size=32)
acc_bce = model_bce.accuracy(X_test, y_test)
print(f"Cross-entropy -> test acc = {acc_bce:.4f}")

plt.figure(figsize=(12, 4))
plt.subplot(1, 2, 1)
plt.plot(model_hinge.train_losses, label='Hinge train')
plt.plot(model_bce.train_losses, label='BCE train')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Сравнение функций потерь')
plt.legend()
plt.grid()

print("\n=== L2-регуляризация ===")
plt.subplot(1, 2, 2)
for lambd in [0.0, 0.001, 0.01, 0.1]:
    m = Perceptron(2)
    m.fit_l2(X_train, y_train, X_val, y_val, epochs=100, lr=0.1, batch_size=32, lambd=lambd)
    acc = m.accuracy(X_test, y_test)
    w_norm = np.sqrt(np.sum(m.w ** 2))
    print(f"λ={lambd:.3f} -> test acc={acc:.4f}, ||w||={w_norm:.4f}")
    plt.plot(m.train_losses, label=f'λ={lambd}')

plt.xlabel('Epoch')
plt.ylabel('Loss + L2')
plt.title('L2-регуляризация')
plt.legend()
plt.grid()
plt.tight_layout()
plt.show()

# ========== Доп. задание 3: Метрики и анализ ошибок ==========
print("\n=== Доп. задание 3: Метрики качества ===")

y_pred = model.predict(X_test)
y_prob = model.forward(X_test).ravel()


def confusion_matrix(y_true, y_pred):
    tp = np.sum((y_true == 1) & (y_pred == 1))
    tn = np.sum((y_true == 0) & (y_pred == 0))
    fp = np.sum((y_true == 0) & (y_pred == 1))
    fn = np.sum((y_true == 1) & (y_pred == 0))
    return tp, tn, fp, fn


def precision(y_true, y_pred):
    tp, _, fp, _ = confusion_matrix(y_true, y_pred)
    return tp / (tp + fp) if (tp + fp) > 0 else 0


def recall(y_true, y_pred):
    tp, _, _, fn = confusion_matrix(y_true, y_pred)
    return tp / (tp + fn) if (tp + fn) > 0 else 0


def f1_score(y_true, y_pred):
    p = precision(y_true, y_pred)
    r = recall(y_true, y_pred)
    return 2 * p * r / (p + r) if (p + r) > 0 else 0


def roc_curve(y_true, y_prob):
    """Строим ROC-кривую вручную"""
    thresholds = np.sort(np.unique(y_prob))[::-1]
    tpr_list, fpr_list = [], []

    for thresh in thresholds:
        y_pred_thresh = (y_prob >= thresh).astype(int)
        tp, tn, fp, fn = confusion_matrix(y_true, y_pred_thresh)
        tpr = tp / (tp + fn) if (tp + fn) > 0 else 0
        fpr = fp / (fp + tn) if (fp + tn) > 0 else 0
        tpr_list.append(tpr)
        fpr_list.append(fpr)

    return np.array(fpr_list), np.array(tpr_list)


def roc_auc(fpr, tpr):
    """Площадь под ROC-кривой (метод трапеций)"""
    return np.trapz(tpr, fpr)


tp, tn, fp, fn = confusion_matrix(y_test, y_pred)
print(f"Confusion matrix: TP={tp}, TN={tn}, FP={fp}, FN={fn}")
print(f"Precision: {precision(y_test, y_pred):.4f}")
print(f"Recall:    {recall(y_test, y_pred):.4f}")
print(f"F1-score:  {f1_score(y_test, y_pred):.4f}")

fpr, tpr = roc_curve(y_test, y_prob)
auc = roc_auc(fpr, tpr)
print(f"ROC-AUC:   {auc:.4f}")

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

axes[0].plot(fpr, tpr, 'b-', lw=2, label=f'ROC (AUC = {auc:.4f})')
axes[0].plot([0, 1], [0, 1], 'k--', lw=1, label='Случайный')
axes[0].fill_between(fpr, tpr, alpha=0.2)
axes[0].set_xlabel('False Positive Rate')
axes[0].set_ylabel('True Positive Rate')
axes[0].set_title('ROC-кривая')
axes[0].legend()
axes[0].grid()

errors = y_pred != y_test
axes[1].scatter(X_test[~errors, 0], X_test[~errors, 1], c=y_test[~errors],
                cmap='bwr', alpha=0.6, edgecolors='k', label='Правильно')
axes[1].scatter(X_test[errors, 0], X_test[errors, 1], c=y_test[errors],
                cmap='bwr', alpha=0.8, edgecolors='yellow', s=100, linewidths=2,
                marker='X', label='Ошибка')
x1_range = np.linspace(X_test[:, 0].min(), X_test[:, 0].max(), 100)
x2_range = -(model.w[0, 0] * x1_range + model.b) / model.w[1, 0]
axes[1].plot(x1_range, x2_range, 'g-', lw=2, label='Граница')
axes[1].set_xlabel('x1')
axes[1].set_ylabel('x2')
axes[1].set_title(f'Ошибки классификации ({errors.sum()} из {len(y_test)})')
axes[1].legend()
axes[1].grid()
plt.tight_layout()
plt.show()

print(f"\nОшибочно классифицировано: {errors.sum()} из {len(y_test)}")
print("Ошибки в основном на точках, близких к разделяющей границе.")

# ========== Доп. задание 4: Momentum ==========
print("\n=== Доп. задание 4: Градиентный спуск с моментом ===")

plt.figure(figsize=(14, 5))

model_sgd = Perceptron(2)
model_sgd.fit(X_train, y_train, X_val, y_val, epochs=100, lr=0.1, batch_size=32)
acc_sgd = model_sgd.accuracy(X_test, y_test)
print(f"SGD (без момента) -> test acc = {acc_sgd:.4f}")
plt.plot(model_sgd.train_losses, 'k--', lw=2, label='SGD (β=0)')

for beta in [0.5, 0.9, 0.99]:
    m = Perceptron(2)
    m.fit_momentum(X_train, y_train, X_val, y_val, epochs=100, lr=0.1, batch_size=32, beta=beta)
    acc = m.accuracy(X_test, y_test)
    print(f"Momentum β={beta:.2f} -> test acc = {acc:.4f}")
    plt.plot(m.train_losses, lw=1.5, label=f'β={beta}')

plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Сравнение SGD и Momentum')
plt.legend()
plt.grid()
plt.tight_layout()
plt.show()

# ========== Доп. задание 5: Кросс-валидация ==========
print("\n=== Доп. задание 5: Кросс-валидация для подбора гиперпараметров ===")


def cross_validate(X, y, lr, batch_size, epochs=50, k=5):
    """k-кратная кросс-валидация"""
    np.random.seed(42)
    n = len(X)
    idx = np.random.permutation(n)
    fold_size = n // k
    accuracies = []

    for fold in range(k):
        val_idx = idx[fold * fold_size: (fold + 1) * fold_size]
        train_idx = np.setdiff1d(idx, val_idx)

        X_train_fold = X[train_idx]
        y_train_fold = y[train_idx].ravel()
        X_val_fold = X[val_idx]
        y_val_fold = y[val_idx].ravel()

        scaler_fold = StandardScaler()
        X_train_fold = scaler_fold.fit_transform(X_train_fold)
        X_val_fold = scaler_fold.transform(X_val_fold)

        model_fold = Perceptron(2)
        model_fold.fit(X_train_fold, y_train_fold, X_val_fold, y_val_fold,
                       epochs=epochs, lr=lr, batch_size=batch_size)

        acc = model_fold.accuracy(X_val_fold, y_val_fold)
        accuracies.append(acc)

    return np.mean(accuracies), np.std(accuracies)

X_cv = np.vstack([X_train, X_val])
y_cv = np.hstack([y_train, y_val]).reshape(-1, 1)

lrs = [0.001, 0.01, 0.1, 0.5]
batch_sizes = [1, 16, 32, 64]

print("Кросс-валидация (5 фолдов):")
print(f"{'lr':<8} {'batch':<8} {'mean acc':<10} {'std':<10}")
print("-" * 40)

best_acc = 0
best_lr = 0.1
best_bs = 32

for lr in lrs:
    for bs in batch_sizes:
        mean_acc, std_acc = cross_validate(X_cv, y_cv, lr, bs, epochs=50, k=5)
        print(f"{lr:<8.3f} {bs:<8} {mean_acc:<10.4f} {std_acc:<10.4f}")
        if mean_acc > best_acc:
            best_acc = mean_acc
            best_lr = lr
            best_bs = bs

print(f"\nЛучшие параметры: lr={best_lr}, batch_size={best_bs}, mean_acc={best_acc:.4f}")

# Финальная модель на ВСЕХ train+val
print(f"\nОбучение финальной модели (lr={best_lr}, batch_size={best_bs})")
X_train_final = np.vstack([X_train, X_val])
y_train_final = np.hstack([y_train, y_val])

model_final = Perceptron(2)
model_final.fit(X_train_final, y_train_final, X_test, y_test, epochs=100,
                lr=best_lr, batch_size=best_bs)
final_acc = model_final.accuracy(X_test, y_test)
print(f"Финальная модель -> test accuracy = {final_acc:.4f}")

plt.figure(figsize=(10, 5))
for lr in lrs:
    means = []
    for bs in batch_sizes:
        mean_acc, _ = cross_validate(X_cv, y_cv, lr, bs, epochs=50, k=5)
        means.append(mean_acc)
    plt.plot(batch_sizes, means, 'o-', lw=2, label=f'lr={lr}')

plt.xscale('log')
plt.xlabel('Batch size')
plt.ylabel('Mean accuracy')
plt.title('Кросс-валидация: выбор гиперпараметров')
plt.legend()
plt.grid()
plt.tight_layout()
plt.show()

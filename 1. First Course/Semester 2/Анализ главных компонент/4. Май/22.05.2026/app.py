from lib import *


st.set_page_config(
	layout="wide",

)

st.title("PCA Examples", text_alignment="center")

decomposition = st.radio("Основная модель", DECOMPOSITION.keys(), horizontal=True)
manifold = st.radio("Дополнительная модель", MANIFOLD.keys(), horizontal=True)
if decomposition == "PCA":
	variance = st.select_slider("Дисперсия (в %)", range(0, 100), value=80, width=600)
dataset = st.radio("Dataset", DATASET.keys(), horizontal=True)

with st.expander("Dataset Info"):
	st.markdown(DATASET[dataset].DESCR, text_alignment="justify")

st.header(f"Methods: {decomposition} + {manifold}. Dataset: {dataset}", text_alignment="center")

if st.button("Начать обучение модели"):
	x, y = DATASET[dataset].data, DATASET[dataset].target
	components = x.shape[1]

	x_scaler = StandardScaler().fit_transform(x)

	progress = st.progress(0, "Обучение модели...")

	model = DECOMPOSITION[decomposition]
	if decomposition == "PCA":
		for i in range(100):
			time.sleep(.05)
			progress.progress(i + 1, "Обучение модели...")

		model.n_components = variance / 100
		model.fit(x_scaler)

	else:
		n_components = x.shape[1]
		model_score = []
		progress_interval = np.linspace(1, 100, n_components)

		for n in range(1 if decomposition == "KernelPCA" else 0, n_components):
			time.sleep(.05)

			if decomposition == "KernelPCA":
				model = DECOMPOSITION[decomposition]

			model.n_components = n

			if decomposition == "KernelPCA":
				model = make_pipeline(model, SVC())
			
			model_score += [cross_val_score(model, x_scaler, y).mean()]
			progress.progress(int(progress_interval[n]), f"Обучение модели... ({n + 1}/{n_components})")

	time.sleep(2)
	progress.empty()

	end = st.badge("Обучение модели завершено", icon=":material/check:", color="green")
	time.sleep(3)
	end.empty()

	fig, ax = plt.subplots()
	if decomposition == "PCA":
		pca = PCA().fit(x_scaler)
		variances = np.cumsum(pca.explained_variance_ratio_)
		model_variance = model.explained_variance_ratio_.sum()
		model_component = model.n_components_
		
		ax.plot(range(1, len(variances) + 1), variances, "o-")
		ax.axhline(model_variance, c="r", ls="--")
		ax.axvline(model_component, c="r", ls="--")
		ax.scatter(model_component, model_variance, s=50, ec="k", zorder=3)
		ax.text(model_component + .5, model_variance - .1, f"var {variance}% - {model_component} components", fontsize=10)
		ax.set_xlabel("Components")
		ax.set_ylabel("Variance")
		ax.set_title("Отношение кол-ва компонент к проценту дисперсии")
		ax.grid(alpha=.5)

	
	else:
		model_component = np.argmax(model_score)
		best_score = model_score[model_component]
		
		ax.plot(range(1 if decomposition == "KernelPCA" else 0, n_components), model_score, "o-")
		ax.axvline(model_component + (1 if decomposition == "KernelPCA" else 0), c="r", ls="--")
		ax.axhline(best_score, c="r", ls="--")
		ax.scatter(model_component + (1 if decomposition == "KernelPCA" else 0), best_score, s=50, ec="k", zorder=3)
		ax.text(model_component + .05, best_score - .005, f"score {best_score:.4f} - {model_component + (1 if decomposition == "KernelPCA" else 0)} components", fontsize=10)
		ax.set_xlabel("Components")
		ax.set_ylabel("Scores")
		ax.set_title("Отношение кол-ва компонент к оценке")
		ax.grid(alpha=.5)

	st.pyplot(fig, width="content")

	fig_2d_3d = plt.figure(figsize=(12, 6))
	ax2d = fig_2d_3d.add_subplot(1, 2, 1)
	ax3d = fig_2d_3d.add_subplot(1, 2, 2, projection="3d")
	ax3d.view_init(elev=10, azim=45)
	ax2d.set_xlabel("Component 1")
	ax2d.set_ylabel("Component 2")
	ax3d.set_xlabel("Component 1")
	ax3d.set_ylabel("Component 2")
	ax3d.set_zlabel("Component 3")

	if 2 <= model_component <= 3:
		fig_2d_3d.suptitle("2D и 3D анализ для основной модели")
		model_2d = DECOMPOSITION[decomposition]
		model_2d.n_components = 2
		model_3d = DECOMPOSITION[decomposition]
		model_3d.n_components = 3

		
	else:
		st.toast(f"Кол-во компонент в основной модели \"{decomposition}\" не соответствует оптимальному кол-ву для визуализации", icon="⚠️", duration="long")
		time.sleep(10)
		st.toast(f"Выполняется обучение дополнительной модели \"{manifold}\". Увидеть результаты можно ниже", icon="🚨")
		fig_2d_3d.suptitle("2D и 3D анализ для дополнительной модели")

		model_2d = MANIFOLD[manifold]
		model_2d.n_components = 2
		model_3d = MANIFOLD[manifold]
		model_3d.n_components = 3

	x_2d = model_2d.fit_transform(x_scaler)
	x_3d = model_3d.fit_transform(x_scaler)

	sc_2d = ax2d.scatter(x_2d[:, 0], x_2d[:, 1], c=y, s=50, ec="k", zorder=3, cmap="viridis")
	ax2d.set_xlabel("Component 1")
	ax2d.set_ylabel("Component 2")
	ax2d.grid(alpha=.5)
	fig_2d_3d.colorbar(sc_2d, ax=ax2d, location="top")

	sc_3d = ax3d.scatter(x_3d[:, 0], x_3d[:, 1], x_3d[:, 2], c=y, s=50, ec="k", cmap="viridis")
	fig_2d_3d.colorbar(sc_3d, ax=ax3d, location="top")

	st.pyplot(fig_2d_3d)
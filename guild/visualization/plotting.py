import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from tqdm import tqdm

from guild.constants.visualization import CONTRAST_PALETTE


def pannel_plot(input_data, usable_columns, save_path=None):
    """
    Plot a pannel of histograms for each feature in usable_columns
    :param input_data: Dataframe with the data
    :param usable_columns: List of columns to plot
    :param save_path: Path to save the plot, if None it will be shown instead
    """
    plt.rc("legend", fontsize=5, title_fontsize=5)
    num_columns = int(len(usable_columns) ** 0.5) + 1
    num_rows = int(len(usable_columns) ** 0.5) + 1

    fig, axs = plt.subplots(num_rows, num_columns, figsize=(num_rows * 3, num_columns * 3))

    variable_count = 0
    for i in range(num_rows):
        for j in range(num_columns):
            if variable_count >= len(usable_columns):
                axs[i, j].axis("off")
                continue

            current_column = usable_columns[variable_count]

            current_subset_no_nan = input_data.dropna(subset=[current_column])

            ax = axs[i, j]

            if current_subset_no_nan["docking"].nunique() != 2:
                print(f"Skipping {current_column} as it has less than 2 categories")
                continue

            # Plot the data and collect handles/labels for the legend
            sns.histplot(
                data=current_subset_no_nan,
                x=current_column,
                kde=True,
                hue="docking",
                common_norm=False,
                stat="density",
                # element="poly",
                alpha=0.75,
                bins=25,
                ax=ax,
                line_kws={"linewidth": 0.5},
                palette=["#16EB96", "#F4B183"],
            )

            ax.tick_params(axis="both", labelsize=8)
            ax.set_title(current_column, fontsize=8)
            ax.set_xlabel("")
            ax.set_ylabel("")
            variable_count += 1

    plt.suptitle("Bulk analysis testing", fontsize=12)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300)
    else:
        plt.show()
    plt.close()


def bulk_plot_unique_scores(input_data, ranks_list, save_folder=None):
    """
    Plot the rank percentile scoring results for unique proteins.
    :param input_data: Dataframe with the data
    :param ranks_list: List of scores to plot
    :param save_folder: Folder to save the plots
    """

    # Scale the counts of each ligand category to the same scale
    for current_score in ranks_list:
        save_path = f"{save_folder}/all_proteins_{current_score}_density_plot.png"
        sns.set(style="whitegrid")
        plt.close()
        plt.figure(figsize=(12, 6))
        input_data = (
            input_data.loc[input_data[current_score] > 0].copy().dropna(subset=[current_score])
        )
        sns.displot(
            input_data.reset_index(drop=True),
            x=current_score,
            hue="ligand_category",
            kind="kde",
            alpha=0.50,
            palette=CONTRAST_PALETTE,
            fill=True,
            common_norm=False,
            warn_singular=False,
        )

        # Set ax-axis to above 0
        plt.title(f"{current_score} Density Plot")
        # plt.xlim(0, 1)
        plt.tight_layout()
        plt.savefig(save_path, dpi=300)

        plt.close()


def bulk_plot_unique_proteins_scorings(input_data, ranks_list, top_n_hits=5, save_folder=None):
    """
    Plot the rank percentile scoring results for unique proteins.
    :param input_data: Dataframe with the data
    :param ranks_list: List of scores to plot
    :param top_n_hits: Number of top hits to plot, to avoid overplotting.
    :param save_folder: Folder to save the plots
    """
    for current_protein in tqdm(
        input_data["protein_config_id"].unique(),
        desc="Plotting unique proteins scorings",
    ):
        current_subset = input_data[input_data["protein_config_id"] == current_protein].copy()
        for current_score in ranks_list:
            comparison_subset = current_subset.loc[
                current_subset["ligand_category"].isin(["decoy", "weak-binder", "strong-binder"])
            ].dropna(subset=[current_score])
            save_path = f"{save_folder}/{current_protein}_{current_score}_density_plot.png"

            # Create figure with extra space at bottom
            plt.close()
            plt.figure(figsize=(12, 8))

            # Main density plot
            plt.gca()
            # Store the density plot to get its legend handles
            density_plot = sns.kdeplot(
                data=comparison_subset.reset_index(drop=True),
                x=current_score,
                hue="ligand_category",
                alpha=0.75,
                palette=CONTRAST_PALETTE,
                fill=True,
                common_norm=False,
                warn_singular=False,
            )

            # Get the legend handles from the density plot
            # density_handles = density_plot.get_legend_handles_labels()[0]
            density_handles = density_plot.legend_.legend_handles
            density_labels = ["Decoy", "Weak Binder", "Strong Binder"]

            # Remove the automatic legend
            density_plot.get_legend().remove()

            # Handle non-decoy methods
            non_decoy_methods = (
                current_subset[
                    ~current_subset["ligand_category"].isin(
                        ["decoy", "weak-binder", "strong-binder"]
                    )
                ]["ligand_category"]
                .unique()
                .tolist()
            )

            # Determine color palette
            if len(non_decoy_methods) > len(CONTRAST_PALETTE):
                palette_to_use = (
                    CONTRAST_PALETTE * (len(non_decoy_methods) // len(CONTRAST_PALETTE))
                    + CONTRAST_PALETTE[: len(non_decoy_methods) % len(CONTRAST_PALETTE)]
                )
            else:
                palette_to_use = CONTRAST_PALETTE[: len(non_decoy_methods)]

            # Calculate x positions and sort methods
            x_positions = []
            for method in non_decoy_methods:
                x_pos = current_subset[current_subset["ligand_category"] == method][
                    current_score
                ].mean()
                x_positions.append(x_pos)

            x_positions_df = pd.DataFrame(
                {"x_positions": x_positions, "methods": non_decoy_methods}
            ).sort_values(by="x_positions", ascending=False)
            x_positions_df = x_positions_df.iloc[:top_n_hits].reset_index(drop=True)

            for x_pos, color in zip(x_positions_df["x_positions"], palette_to_use, strict=True):
                plt.axvline(x=x_pos, color=color, linestyle="--")

            plt.title(f"{current_protein}, {current_score} Density Plot")

            # Create legend elements for methods
            method_elements = []
            for method, color in zip(x_positions_df["methods"], palette_to_use, strict=True):
                method_elements.append(
                    plt.Rectangle((0, 0), 1, 1, fc=color, alpha=0.5, label=method)
                )

            # Combine both sets of legend elements
            all_handles = density_handles + method_elements

            all_labels = density_labels + x_positions_df["methods"].tolist()
            # Calculate number of columns for combined legend
            n_total = len(all_labels)
            n_cols = min(3, n_total)  # Max 3 columns
            n_rows = (n_total + n_cols - 1) // n_cols

            # Add combined legend below plot
            plt.legend(
                handles=all_handles,
                labels=all_labels,
                loc="upper center",
                bbox_to_anchor=(0.5, -0.2),
                ncol=n_cols,
                frameon=True,
                fancybox=True,
                shadow=True,
                title="Categories and Methods",
                title_fontsize=10,
            )

            plt.tight_layout()
            # Adjust bottom margin to make room for legend
            plt.subplots_adjust(bottom=0.2 + (0.05 * n_rows))
            if save_folder:
                plt.savefig(save_path, dpi=300, bbox_inches="tight")
            else:
                plt.show()
            plt.close()

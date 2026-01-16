import itertools
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
# custom_lines = [
#     Line2D([0], [0], color='blue', lw=2, label='Sine wave'),
#     Line2D([0], [0], color='red', lw=2, linestyle='--', label='Cosine wave')
# ]
legend_handles = []

path = "/Users/ksitterl/Documents/Python/pfas_sensitivities/pfas_sensitivities"
inputs_file = "/Users/ksitterl/Documents/Python/pfas_sensitivities/pfas_sensitivities/data/ix_case_study_sensitivity_inputs.csv"
ix_inputs = pd.read_csv(inputs_file)
ix_nums = [18, 13, 304]
gac_nums = [0, 14, 21]
leg_dict = dict(
    ebct="EBCT",
    loading_rate="Loading Rate",
    resin_cost="Resin Cost",
    # freundlich_k="Freundlich K",
    freundlich_k="K$_F$",
    freundlich_ninv="Freundlich 1/n",
    # surf_diff_coeff="Surface Diffusion Coefficient",
    surf_diff_coeff="D$_s$",
)
title_dict = dict(
    LCOW="LCOW",
    t_breakthru_day="Breakthrough Time",
    total_operating_cost="Total Operating Cost",
)

regen_tag1 = "ethanol_nacl"
regen_tag2 = "methanol_nacl"
regen_tag3 = "acetone_nacl"
# for y_col in ["LCOW", "t_breakthru_day", "total_operating_cost"]:
for y_col in ["LCOW"]:
    figsize = (6, 4)
    figsize = (10, 6)
    fig, ax = plt.subplots(figsize=figsize)
    # fig2, ax2 = plt.subplots(figsize=figsize)

    all_xs = []
    all_labels = []
    # xlabels = []
    # ys = []
    # stds = []
    sweep = "ebct"
    n = 0
    colors = itertools.cycle(["r", "g", "b", "m", "c"])
    # y_col = "LCOW"
    # y_col = "t_breakthru_day"
    nstd = 1
    # for sweep in ["ebct", "loading_rate", "resin_cost", "freundlich_k", "freundlich_ninv", "surf_diff_coeff"]:
    for sweep in [
        "ebct",
        "loading_rate",
        "resin_cost",
        "freundlich_k",
        "surf_diff_coeff",
    ]:
        # for sweep in ["ebct",]:

        xs1 = []
        xlabels1 = []
        ys1 = []
        stds1 = [[], []]

        xs2 = []
        xlabels2 = []
        ys2 = []
        stds2 = [[], []]
        
        xs3 = []
        xlabels3 = []
        ys3 = []
        stds3 = [[], []]

        xs4 = []
        xlabels4 = []
        ys4 = []
        stds4 = [[], []]
        # c = next(colors)

        for i, data in ix_inputs.iterrows():

            if data.curve_id not in ix_nums:
                continue
            try:
                save_file = f"{path}/results/ix/ix_pfas_{sweep}_sensitivity-{data.ref}_curve{data.curve_id}_{data.target_component}_{data.resin}_single_use.csv"
                data1 = pd.read_csv(save_file)
                save_file = f"{path}/results/ix/ix_pfas_{sweep}_sensitivity-{data.ref}_curve{data.curve_id}_{data.target_component}_{data.resin}_{regen_tag1}.csv"
                data2 = pd.read_csv(save_file)
                save_file = f"{path}/results/ix/ix_pfas_{sweep}_sensitivity-{data.ref}_curve{data.curve_id}_{data.target_component}_{data.resin}_{regen_tag2}.csv"
                data3 = pd.read_csv(save_file)
                save_file = f"{path}/results/ix/ix_pfas_{sweep}_sensitivity-{data.ref}_curve{data.curve_id}_{data.target_component}_{data.resin}_{regen_tag3}.csv"
                data4 = pd.read_csv(save_file)
            except:
                print(
                    f"FAILED curve {data.curve_id} of {len(ix_inputs)}: {data.ref}, {data.target_component}, {data.resin}"
                )
                assert False
            
            n += 2

            base_row1 = data1.iloc[-1]
            df1 = data1.iloc[:-1]  # remove base case

            base_row2 = data2.iloc[-1]
            df2 = data2.iloc[:-1]  # remove base case

            base_row3 = data3.iloc[-1]
            df3 = data3.iloc[:-1]  # remove base case

            base_row4 = data4.iloc[-1]
            df4 = data4.iloc[:-1]  # remove base case

            base_y1 = base_row1[y_col]
            base_y2 = base_row2[y_col]
            base_y3 = base_row3[y_col]
            base_y4 = base_row4[y_col]

            print(f"Base {y_col}1: {base_y1}")
            print(f"Base {y_col}2: {base_y2}")
            print(f"Base {y_col}3: {base_y3}")
            print(f"Base {y_col}4: {base_y4}\n")

            # df_nonopt1 = df1[df1.optimal_solve != 1].copy()
            df_opt1 = df1[df1.optimal_solve == 1].copy()
            df_opt1 = df_opt1[df_opt1.t_breakthru_day >= 7].copy()

            # df_nonopt2 = df2[df2.optimal_solve != 1].copy()
            df_opt2 = df2[df2.optimal_solve == 1].copy()
            df_opt2 = df_opt2[df_opt2.t_breakthru_day >= 7].copy()

            # df_nonopt2 = df2[df2.optimal_solve != 1].copy()
            df_opt3 = df3[df3.optimal_solve == 1].copy()
            df_opt3 = df_opt3[df_opt3.t_breakthru_day >= 7].copy()

            # df_nonopt2 = df2[df2.optimal_solve != 1].copy()
            df_opt4 = df4[df4.optimal_solve == 1].copy()
            df_opt4 = df_opt4[df_opt4.t_breakthru_day >= 7].copy()

            if df_opt1.empty:
                print("No optimal results")
                # n -= 3
                continue
            if df_opt2.empty:
                print("No optimal results")
                # n -= 3
                continue
            if df_opt3.empty:
                print("No optimal results")
                # n -= 3
                continue
            lab = f"{data.ref.replace('_', ' ').title()}-{data.target_component}-{data.resin.upper()}"


            ydata1 = df_opt1[y_col].values
            ydata_rel1 = ydata1 / base_y1 - 1

            ydata2 = df_opt2[y_col].values
            ydata_rel2 = ydata2 / base_y2 - 1

            ydata3 = df_opt3[y_col].values
            ydata_rel3 = ydata3 / base_y3 - 1

            ydata4 = df_opt4[y_col].values
            ydata_rel4 = ydata4 / base_y4 - 1

            all_labels.append(lab)
            all_xs.append(n + 1.5)

            stds1[0].append(abs(base_y1 - ydata1.min()))
            stds1[1].append(abs(base_y1 - ydata1.max()))
            xs1.append(n)
            xlabels1.append(lab)
            ys1.append(base_y1)
            
            n += 1

            stds2[0].append(abs(base_y2 - ydata2.min()))
            stds2[1].append(abs(base_y2 - ydata2.max()))
            xs2.append(n)
            xlabels2.append(lab)
            ys2.append(base_y2)
            
            n += 1

            stds3[0].append(abs(base_y3 - ydata3.min()))
            stds3[1].append(abs(base_y3 - ydata3.max()))
            xs3.append(n)
            xlabels3.append(lab)
            ys3.append(base_y3)

            n += 1

            stds4[0].append(abs(base_y4 - ydata4.min()))
            stds4[1].append(abs(base_y4 - ydata4.max()))
            xs4.append(n)
            xlabels4.append(lab)
            ys4.append(base_y4)
            # ax.vlines(n + 0.25, 0, 10)

        # ax.vlines(n + 0.75, 0, 10)
        n += int(len(xlabels1) / 2) + 1.5  # add space between sweeps
        c = next(colors)
        ax.errorbar(
            xs1, ys1, yerr=stds1, fmt=".", capsize=3, color=c, label=leg_dict[sweep]
        )

        eb2 = ax.errorbar(
            # xs2, ys2, yerr=stds2, fmt=".", capsize=3, color=c, label=leg_dict[sweep] + " w/ Regen"
            xs2,
            ys2,
            yerr=stds2,
            fmt=".",
            capsize=3,
            color=c,
        )
        eb2[-1][0].set_linestyle(":")


        eb3 = ax.errorbar(
            # xs2, ys2, yerr=stds2, fmt=".", capsize=3, color=c, label=leg_dict[sweep] + " w/ Regen"
            xs3,
            ys3,
            yerr=stds3,
            fmt=".",
            capsize=3,
            color=c,
        )
        eb3[-1][0].set_linestyle("--")

        eb4 = ax.errorbar(
            # xs2, ys2, yerr=stds2, fmt=".", capsize=3, color=c, label=leg_dict[sweep] + " w/ Regen"
            xs4,
            ys4,
            yerr=stds4,
            fmt=".",
            capsize=3,
            color=c,
        )
        eb4[-1][0].set_linestyle("-.")

        # ax2.errorbar(
        #     xs2, ys2, yerr=stds2, fmt=".", capsize=3, color=c, label=leg_dict[sweep]
        # )
        # ax.set_xticks(_xs)

        # ax.set_xticklabels(xlabels, rotation=30, ha='right')
    fontsize = 12

    # ax.legend(loc="upper left", bbox_to_anchor=(1.01, 1))
    # ax.grid(visible=True
    ax.set_xticks(all_xs)
    ax.set_xticklabels(all_labels, rotation=45, ha="right", fontsize=fontsize)

    ax.set_ylabel(f"{title_dict[y_col]}\nMin/Max", fontsize=fontsize)
    fig.supxlabel("Case Study - Species - Resin", y=-0.04)
    fig.suptitle(
        f"Ion Exchange PFAS Sensitivity Analysis\n{title_dict[y_col]}",
        fontsize=fontsize + 2,
    )
    fig.tight_layout()
# ax.legend(loc="upper left", bbox_to_anchor=(1.01, 1))
hands, labes = ax.get_legend_handles_labels()
# ax.legend(handles=hands, loc="upper left", bbox_to_anchor=(1.01, 1))
# legs.append("test1")
# legs.append("test2")
hands.append(Line2D([0], [0], color='k', lw=2, linestyle='-', label='Single-use'))
hands.append(Line2D([0], [0], color='k', lw=2, linestyle=':', label='Ethanol/NaCl Regen'))
hands.append(Line2D([0], [0], color='k', lw=2, linestyle='--', label='Methanol/NaCl Regen'))
hands.append(Line2D([0], [0], color='k', lw=2, linestyle='-.', label='Acetone/NaCl Regen'))
ax.legend(handles=hands, loc="upper left", bbox_to_anchor=(1.01, 1))
# fig2.supxlabel("Case Study - Species - Resin", y=-0.
# print(len(legs), len(hands))
plt.show()
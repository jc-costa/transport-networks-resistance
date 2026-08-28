def plot_node_number(N, intersections_points, all_connected_nodes, all_connected_pairs):
    # Define custom colors
    line_color = 'steelblue'
    marker_color = 'steelblue'
    line_color_1 = 'red'

    # Define marker style for data points
    marker_style = 'o'
    marker_size = 6

    # Set larger figure size
    plt.figure(figsize=(12, 10))

    # Unzip the intersections_points array to separate x and y values
    x_values, y_values = zip(*intersections_points)

    # Plot all points with their numbers from all_connected_nodes
    for i, connected_nodes in enumerate(all_connected_nodes):
        for node in connected_nodes:
            # Plot points
            plt.scatter(x_values[node], y_values[node], c=marker_color, marker=marker_style, label='Connected Points', s=marker_size, zorder=2)
            # Annotate the point with its number
            plt.annotate(str(node), (x_values[node], y_values[node]), textcoords="offset points", xytext=(0,5), ha='center', fontsize=8, zorder=3)

    # Plot lines between connected pairs
    for connected_pairs in all_connected_pairs:
        for pair in connected_pairs:
            i, j = pair
            plt.plot([x_values[i], x_values[j]], [y_values[i], y_values[j]], linestyle='-', color=line_color, zorder=0)

    # Set title
    plt.title(f'N = {N}', fontsize=22)

    # Adjust plot limits
    #plt.xlim(-0.3, 0.7)
    #plt.ylim(-0.3, 0.7)

    # Remove x and y axis values (ticks and labels)
    plt.xticks([])
    plt.yticks([])

    # Save the plot as an image file
    plt.savefig(f'N_{N}_just_the_paths.png')

    # Display the plot
    plt.show()



def plot_filtered_lines_with_numbers_2(N, intersections_points, filtered_connected_nodes, filtered_connected_pairs, filtered_connected_lines):
    # Define custom colors
    node_color = 'steelblue'
    line_color = 'red'

    # Define marker style for data points
    marker_style = 'o'
    marker_size = 6

    # Set larger figure size
    plt.figure(figsize=(12, 10))

    # Unzip the intersections_points array to separate x and y values
    x_values, y_values = zip(*intersections_points)

    # Plot all nodes from filtered_connected_nodes
    for node_group in filtered_connected_nodes:
        for node in node_group:
            plt.scatter(x_values[node], y_values[node], c=node_color, marker=marker_style, s=marker_size, zorder=2)

    # Plot lines between filtered pairs and annotate with line numbers
    for pair_group, line_group in zip(filtered_connected_pairs, filtered_connected_lines):
        line_number = 0  # Reset line number for each group
        for (i, j) in pair_group:
            # Plot the line connecting nodes i and j
            plt.plot([x_values[i], x_values[j]], [y_values[i], y_values[j]], linestyle='-', color=line_color, zorder=1)

            # Calculate the midpoint of the line
            mid_x = (x_values[i] + x_values[j]) / 2
            mid_y = (y_values[i] + y_values[j]) / 2

            # Adjust the annotation position to avoid overlap
            # Use a small offset based on the line number
            offset_x = 0.02* (line_number % 3)  # Adjust x offset
            offset_y = 0.02 * ((line_number // 3) % 3)  # Adjust y offset

            # Annotate the line number with the offset
            plt.annotate(
                str(line_number),
                (mid_x + offset_x, mid_y + offset_y),  # Apply offset
                textcoords="offset points",
                xytext=(0, 5),  # Additional offset for text
                ha='center',
                fontsize=8,
                zorder=3,
                color='black'  # Use a contrasting color for readability
            )

            # Increment the line number
            line_number += 1

    # Set title
    plt.title(f'N = {N}', fontsize=22)

    # Adjust plot limits
    plt.xlim(-0.3, 1.3)
    plt.ylim(-0.5, 1.3)

    # Remove x and y axis values (ticks and labels)
    plt.xticks([])
    plt.yticks([])

    # Save the plot as an image file
    plt.savefig(f'N_{N}_filtered_lines_with_numbers.png')

    # Display the plot
    plt.show()
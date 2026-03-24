import gradio as gr
from layers import add_layer, update_layer, delete_layer, reset_layers, layer_map, insert_layer
from app_training_client import remote_train, remote_predict

# --------------------------------
# Gradio wiring
# --------------------------------

mobile_css = """
/* Mobile-first layout adjustments */
.gradio-container {
    max-width: 600px !important;
    margin: auto !important;
    border-radius: 12px;
}
/* By default Gradio handles the dark/light toggle natively in the top right */
"""

with gr.Blocks(
    title="NeuralNT Mobile App",
    theme=gr.themes.Default(primary_hue="blue", neutral_hue="slate"),
    css=mobile_css
) as dashboard:

    with gr.Tab("Build Model"):
        gr.Markdown("## Build a Layer")
        builder_arch = gr.Textbox(label="Architecture So Far", lines=6)

        layer_type_dropdown = gr.Dropdown(choices=list(layer_map.keys()), label="Layer Type", value="Linear")
        in_dim = gr.Textbox(label="Input Dim (Linear/Conv2d)")
        out_dim = gr.Textbox(label="Output Dim (Linear/Conv2d)")

        conv_kernel = gr.Textbox(label="Kernel Size", value="3", visible=False)
        conv_padding = gr.Textbox(label="Padding", value="1", visible=False)
        conv_stride = gr.Textbox(label="Stride", value="1", visible=False)
        conv_bias = gr.Checkbox(label="Include Bias", value=True, visible=False)

        pool_kernel = gr.Textbox(label="Pool Kernel Size", value="2", visible=False)
        pool_stride = gr.Textbox(label="Stride", value="2", visible=False)
        pool_padding = gr.Textbox(label="Padding", value="0", visible=False)

        avgpool_kernel = gr.Textbox(label="AvgPool Kernel Size", value="2", visible=False)
        avgpool_stride = gr.Textbox(label="Stride", value="2", visible=False)
        avgpool_padding = gr.Textbox(label="Padding", value="0", visible=False)

        leaky_relu_slope = gr.Textbox(label="Negative Slope", value="0.01", visible=False)
        elu_alpha = gr.Textbox(label="ELU Alpha", value="1.0", visible=False)

        add_btn = gr.Button("Add Layer")
        add_btn.click(
            fn=add_layer,
            inputs=[
                layer_type_dropdown, in_dim, out_dim,
                conv_kernel, conv_padding, conv_stride, conv_bias,
                pool_kernel, pool_stride, pool_padding,
                avgpool_kernel, avgpool_stride, avgpool_padding,
                leaky_relu_slope, elu_alpha
            ],
            outputs=builder_arch
        )

        def toggle_fields(layer_type):
            is_conv = (layer_type == "Conv2d")
            is_pool = (layer_type == "MaxPool2d")
            is_avgpool = (layer_type == "AvgPool2d")
            is_leaky = (layer_type == "LeakyReLU")
            is_elu = (layer_type == "ELU")
            return [
                gr.update(visible=is_conv),
                gr.update(visible=is_conv),
                gr.update(visible=is_conv),
                gr.update(visible=is_conv),
                gr.update(visible=is_pool),
                gr.update(visible=is_pool),
                gr.update(visible=is_pool),
                gr.update(visible=is_avgpool),
                gr.update(visible=is_avgpool),
                gr.update(visible=is_avgpool),
                gr.update(visible=is_leaky),
                gr.update(visible=is_elu),
            ]

        layer_type_dropdown.change(
            toggle_fields,
            inputs=[layer_type_dropdown],
            outputs=[
                conv_kernel, conv_padding, conv_stride, conv_bias,
                pool_kernel, pool_stride, pool_padding,
                avgpool_kernel, avgpool_stride, avgpool_padding,
                leaky_relu_slope, elu_alpha
            ]
        )

        reset_btn = gr.Button("Reset Layers")
        reset_btn.click(fn=reset_layers, inputs=[], outputs=builder_arch)

        gr.Markdown("### Edit or Delete a Layer")
        layer_index = gr.Number(label="Layer Index (0-based)", precision=0)
        new_layer_type = gr.Dropdown(list(layer_map.keys()), label="New Layer Type")
        new_in_dim = gr.Textbox(label="New Input Dim (if applicable)")
        new_out_dim = gr.Textbox(label="New Output Dim (if applicable)")

        edit_kernel = gr.Textbox(label="Kernel Size", value="3", visible=False)
        edit_padding = gr.Textbox(label="Padding", value="1", visible=False)
        edit_stride = gr.Textbox(label="Stride", value="1", visible=False)
        edit_bias = gr.Checkbox(label="Include Bias", value=True, visible=False)

        edit_pool_kernel = gr.Textbox(label="Pool Kernel Size", value="2", visible=False)
        edit_pool_stride = gr.Textbox(label="Stride", value="2", visible=False)
        edit_pool_padding = gr.Textbox(label="Padding", value="0", visible=False)

        edit_avgpool_kernel = gr.Textbox(label="AvgPool Kernel Size", value="2", visible=False)
        edit_avgpool_stride = gr.Textbox(label="Stride", value="2", visible=False)
        edit_avgpool_padding = gr.Textbox(label="Padding", value="0", visible=False)

        edit_leaky_relu_slope = gr.Textbox(label="Negative Slope", value="0.01", visible=False)
        edit_elu_alpha = gr.Textbox(label="ELU Alpha", value="1.0", visible=False)

        edit_btn = gr.Button("Edit Layer")
        delete_btn = gr.Button("Delete Layer")
        insert_btn = gr.Button("Insert New Layer")

        edit_btn.click(
            fn=update_layer,
            inputs=[
                layer_index, new_layer_type, new_in_dim, new_out_dim,
                edit_kernel, edit_padding, edit_stride, edit_bias,
                edit_pool_kernel, edit_pool_stride, edit_pool_padding,
                edit_avgpool_kernel, edit_avgpool_stride, edit_avgpool_padding,
                edit_leaky_relu_slope, edit_elu_alpha
            ],
            outputs=builder_arch
        )

        delete_btn.click(fn=delete_layer, inputs=[layer_index], outputs=builder_arch)
        insert_btn.click(
            fn=insert_layer,
            inputs=[
                layer_index, new_layer_type, new_in_dim, new_out_dim,
                edit_kernel, edit_padding, edit_stride, edit_bias,
                edit_pool_kernel, edit_pool_stride, edit_pool_padding,
                edit_avgpool_kernel, edit_avgpool_stride, edit_avgpool_padding,
                edit_leaky_relu_slope, edit_elu_alpha
            ],
            outputs=builder_arch
        )

        def toggle_edit_fields(layer_type):
            is_conv = (layer_type == "Conv2d")
            is_pool = (layer_type == "MaxPool2d")
            is_avgpool = (layer_type == "AvgPool2d")
            is_leaky = (layer_type == "LeakyReLU")
            is_elu = (layer_type == "ELU")
            return [
                gr.update(visible=is_conv),
                gr.update(visible=is_conv),
                gr.update(visible=is_conv),
                gr.update(visible=is_conv),
                gr.update(visible=is_pool),
                gr.update(visible=is_pool),
                gr.update(visible=is_pool),
                gr.update(visible=is_avgpool),
                gr.update(visible=is_avgpool),
                gr.update(visible=is_avgpool),
                gr.update(visible=is_leaky),
                gr.update(visible=is_elu),
            ]

        new_layer_type.change(
            toggle_edit_fields,
            inputs=[new_layer_type],
            outputs=[
                edit_kernel, edit_padding, edit_stride, edit_bias,
                edit_pool_kernel, edit_pool_stride, edit_pool_padding,
                edit_avgpool_kernel, edit_avgpool_stride, edit_avgpool_padding,
                edit_leaky_relu_slope, edit_elu_alpha
            ]
        )

    # ── Train Tab ─────────────────────────────────────────────────────────
    with gr.Tab("Train"):
        gr.Markdown(
            "## Train the Model ☁️\n"
            "> Training runs on the **HuggingFace cloud GPU**. No local GPU is required."
        )

        loss_dropdown    = gr.Dropdown(['MSELoss', 'CrossEntropyLoss'], label='Loss Function', value='CrossEntropyLoss')
        opt_dropdown     = gr.Dropdown(['SGD', 'Adam'], label='Optimizer', value='Adam')
        lr_box           = gr.Textbox(value="0.01", label="Learning Rate")
        batch_box        = gr.Textbox(value="32",   label="Batch Size")
        size_box         = gr.Textbox(value="32",   label="Image Resize (e.g. 32x32)")
        file_box         = gr.File(label="Upload Dataset (CSV or ZIP)", type="filepath")
        custom_box       = gr.Textbox(label="Custom Extraction Path (optional)")
        epochs_box       = gr.Textbox(value="10",  label="Epochs")
        generate_3d_checkbox     = gr.Checkbox(
            label="Generate 3D Descent Animation (⚠️ Slower, GPU/RAM-intensive)", value=False)
        generate_3d_targetframes = gr.Textbox(value="300", label="Target Frames for Video", visible=False)
        generate_3d_framerate    = gr.Textbox(value="10",  label="Frame Rate (Frames per Second)", visible=False)
        channel_dropdown         = gr.Dropdown(
            [1, 3], label="Input Channels (1 = Grayscale, 3 = RGB)", value=3)

        # Outputs (same as before)
        loss_curve      = gr.Image(label="Loss Curve")
        animation_video = gr.Video(label="3D Descent Animation")
        model_file      = gr.File(label="Download Trained Model")
        log_box         = gr.Markdown(label="Log")
        
        with gr.Row():
            train_button = gr.Button("🚀 Start Cloud Training")
            stop_button  = gr.Button("⏹ Stop Local Stream")

    # ── Predict Tab ───────────────────────────────────────────────────────
    import os, glob
    def get_history():
        if not os.path.exists("trained_models"):
            return []
        return sorted(glob.glob("trained_models/*.pt"), reverse=True)

    with gr.Tab("Predict 🔮"):
        gr.Markdown(
            "## Run Inference Online\n"
            "> Predict using models safely cached in history, or manually upload a `.pt` file!"
        )
        
        with gr.Row():
            pred_history_dropdown = gr.Dropdown(choices=get_history(), label="1. Select Model From History (.pt)")
            refresh_history_btn = gr.Button("🔄 Refresh History", scale=0)
            
        gr.Markdown("**OR**")
        pred_model_file = gr.File(label="1b. Upload specific Trained Model (.pt)", type="filepath")
        
        with gr.Row():
            pred_image_file = gr.Image(label="2. Image to Predict", type="filepath")
            pred_tabular_data = gr.Textbox(label="OR Tabular Data (comma separated)", placeholder="1.2, 3.4, 5.6")
            
        with gr.Row():
            pred_image_size = gr.Textbox(value="32", label="Image Resize Expected (e.g. 32)")
            pred_num_channels = gr.Dropdown([1, 3], label="Input Channels (1=Gray, 3=RGB)", value=3)
            
        pred_class_names = gr.Textbox(
            label="Class Names Mapping (optional, comma-separated)", 
            value="airplane, automobile, bird, cat, deer, dog, frog, horse, ship, truck"
        )
            
        predict_button = gr.Button("🔮 Run Prediction Natively")
        predict_output = gr.Markdown(label="Result Log")
        
        refresh_history_btn.click(fn=lambda: gr.Dropdown(choices=get_history()), inputs=None, outputs=pred_history_dropdown)

    # ── Wire up buttons ───────────────
    train_click = train_button.click(
        fn=remote_train,
        inputs=[
            loss_dropdown,
            opt_dropdown,
            lr_box,
            batch_box,
            size_box,
            file_box,
            custom_box,
            epochs_box,
            channel_dropdown,
            generate_3d_checkbox,
            generate_3d_targetframes,
            generate_3d_framerate,
        ],
        outputs=[
            loss_curve,
            animation_video,
            model_file,
            builder_arch,
            log_box,
        ]
    )
    
    stop_button.click(fn=None, inputs=None, outputs=None, cancels=[train_click])

    def predict_wrapper(hist_path, file_path, img_path, tab_data, img_size, c_channels, c_names):
        actual_model = file_path if file_path else hist_path
        return remote_predict(actual_model, img_path, tab_data, img_size, c_channels, c_names)

    predict_button.click(
        fn=predict_wrapper,
        inputs=[
            pred_history_dropdown,
            pred_model_file,
            pred_image_file,
            pred_tabular_data,
            pred_image_size,
            pred_num_channels,
            pred_class_names
        ],
        outputs=predict_output
    )

dashboard.queue()
dashboard.launch(share=True)
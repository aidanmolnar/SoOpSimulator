module.exports = {
    module: {
        rules: [
            {
                test: /\.vtkjs$/,
                use: [
                    {
                        loader: 'url-loader',
                        options: {
                            esModule: false,
                        },
                    },
                ],
            },
        ],
    },
};